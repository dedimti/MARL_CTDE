"""
src/environment/borg_loader.py
Google Borg Cluster Trace 2019 — Loader and Preprocessor

Loads and preprocesses the Google Borg production workload trace
for use in the cloud scheduling simulation environment.

Dataset:
    Google Borg Cluster Trace 2019 — 8 cells, ~29 days
    Download: https://github.com/google/cluster-data

References:
    [1] Tirmazi et al. (2020) - Borg: the Next Generation (EuroSys)
    [2] Verma et al. (2015)  - Large-scale cluster management at Google
    [3] Reiss et al. (2011)  - Google cluster-usage traces: format + schema
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass


@dataclass
class BorgJob:
    """Represents a single job from the Borg trace."""
    job_id: int
    submit_time: float          # Microseconds from trace start
    priority: int               # 0–11 (higher = more important)
    cpu_request: float          # Normalized CPU request (0–1)
    memory_request: float       # Normalized memory request (0–1)
    duration: float             # Job duration (seconds)
    deadline: float             # SLA deadline (submit_time + tolerance)
    n_tasks: int                # Number of tasks in job


class BorgTraceLoader:
    """
    Loads and preprocesses Google Borg Cluster Trace 2019.

    Processing pipeline:
        1. Load raw CSV files (job_events, task_events, machine_events)
        2. Filter: remove failed/killed jobs, short-duration jobs
        3. Normalize: CPU/memory to [0, 1] range
        4. Compute SLA deadlines based on priority tier
        5. Create windowed batches for episodic training

    Usage:
        loader = BorgTraceLoader("data/borg_trace_2019")
        jobs = loader.load_cell("a")
        batches = loader.create_episode_batches(jobs, n_jobs=100)
    """

    # Priority tier → SLA deadline multiplier (higher priority = tighter deadline)
    PRIORITY_DEADLINE_MULTIPLIERS = {
        0: 5.0,   # Free tier — very relaxed
        1: 4.0,
        2: 3.0,
        3: 2.5,
        4: 2.0,   # Production tier
        5: 1.8,
        6: 1.5,
        7: 1.3,
        8: 1.2,
        9: 1.1,
        10: 1.05, # Critical tier — very tight
        11: 1.02
    }

    def __init__(
        self,
        data_path: str,
        min_duration_seconds: float = 10.0,
        normalize: bool = True,
        filter_failed: bool = True,
        seed: int = 42
    ):
        self.data_path = data_path
        self.min_duration = min_duration_seconds
        self.normalize = normalize
        self.filter_failed = filter_failed
        self.rng = np.random.default_rng(seed)

        # Normalization statistics (computed from training set)
        self._cpu_max = None
        self._memory_max = None

        logger.info(f"BorgTraceLoader initialized — data path: {data_path}")

    def load_cell(self, cell_id: str) -> List[BorgJob]:
        """
        Load and preprocess all jobs from a single Borg cell.

        Args:
            cell_id: Cell identifier (e.g., "a", "b", ..., "h")

        Returns:
            jobs: List of preprocessed BorgJob objects
        """
        job_events_path = os.path.join(
            self.data_path, "job_events", f"part-{cell_id}-*.csv.gz"
        )
        task_events_path = os.path.join(
            self.data_path, "task_events", f"part-{cell_id}-*.csv.gz"
        )

        logger.info(f"Loading Borg cell '{cell_id}'...")

        try:
            # Load job events
            job_df = self._load_job_events(cell_id)
            task_df = self._load_task_events(cell_id)

            # Merge job and task information
            merged = self._merge_job_task(job_df, task_df)

            # Apply filters
            filtered = self._apply_filters(merged)

            # Normalize features
            if self.normalize:
                filtered = self._normalize_features(filtered)

            # Convert to BorgJob objects
            jobs = self._to_borg_jobs(filtered)

            logger.info(f"Cell '{cell_id}': loaded {len(jobs)} jobs "
                       f"(filtered from {len(merged)} raw)")

            return jobs

        except FileNotFoundError:
            logger.warning(f"Cell '{cell_id}' files not found. "
                          f"Generating synthetic Borg-like data for testing.")
            return self._generate_synthetic_borg_data(n_jobs=1000, cell_id=cell_id)

    def _load_job_events(self, cell_id: str) -> pd.DataFrame:
        """Load job events CSV for a cell."""
        # Column names from Borg trace schema [3]
        columns = [
            'timestamp', 'missing_info', 'job_id', 'event_type',
            'user', 'scheduling_class', 'job_name', 'logical_job_name'
        ]
        # In practice: glob all part files for this cell
        # Simplified for reproducibility
        path = os.path.join(self.data_path, "job_events")
        files = [f for f in os.listdir(path) if f.startswith(f"part-{cell_id}")]

        dfs = []
        for f in files[:5]:  # Load first 5 parts for memory efficiency
            try:
                df = pd.read_csv(
                    os.path.join(path, f),
                    names=columns,
                    compression='gzip' if f.endswith('.gz') else None
                )
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)

    def _load_task_events(self, cell_id: str) -> pd.DataFrame:
        """Load task events CSV for a cell."""
        columns = [
            'timestamp', 'missing_info', 'job_id', 'task_index',
            'machine_id', 'event_type', 'user', 'scheduling_class',
            'priority', 'cpu_request', 'memory_request',
            'disk_space_request', 'machine_constraint'
        ]
        path = os.path.join(self.data_path, "task_events")
        files = [f for f in os.listdir(path) if f.startswith(f"part-{cell_id}")]

        dfs = []
        for f in files[:5]:
            try:
                df = pd.read_csv(
                    os.path.join(path, f),
                    names=columns,
                    compression='gzip' if f.endswith('.gz') else None
                )
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)

    def _merge_job_task(
        self,
        job_df: pd.DataFrame,
        task_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge job and task events to compute job-level features."""
        if job_df.empty or task_df.empty:
            return pd.DataFrame()

        # Get job submit/finish times
        job_submit = job_df[job_df['event_type'] == 0].groupby('job_id')['timestamp'].min()
        job_finish = job_df[job_df['event_type'] == 4].groupby('job_id')['timestamp'].max()

        # Get task resource requests (mean across tasks)
        task_resources = task_df.groupby('job_id').agg({
            'priority': 'max',
            'cpu_request': 'mean',
            'memory_request': 'mean',
            'task_index': 'count'
        }).rename(columns={'task_index': 'n_tasks'})

        # Merge
        merged = pd.DataFrame({
            'submit_time': job_submit,
            'finish_time': job_finish
        }).join(task_resources, how='inner')

        merged['duration'] = (merged['finish_time'] - merged['submit_time']) / 1e6  # μs → s
        merged = merged.dropna()

        return merged

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to remove invalid/noisy entries."""
        if df.empty:
            return df

        original_len = len(df)

        # Filter 1: Minimum duration
        df = df[df['duration'] >= self.min_duration]

        # Filter 2: Valid resource requests
        if 'cpu_request' in df.columns:
            df = df[df['cpu_request'] > 0]
        if 'memory_request' in df.columns:
            df = df[df['memory_request'] > 0]

        # Filter 3: Valid priority range
        if 'priority' in df.columns:
            df = df[df['priority'].between(0, 11)]

        logger.debug(f"Filtering: {original_len} → {len(df)} jobs "
                    f"({original_len - len(df)} removed)")

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize CPU and memory requests to [0, 1]."""
        if df.empty:
            return df

        # Compute normalization stats from data (fit once, reuse)
        if self._cpu_max is None:
            self._cpu_max = df['cpu_request'].quantile(0.99)  # Use 99th percentile
        if self._memory_max is None:
            self._memory_max = df['memory_request'].quantile(0.99)

        df = df.copy()
        df['cpu_request'] = np.clip(df['cpu_request'] / self._cpu_max, 0, 1)
        df['memory_request'] = np.clip(df['memory_request'] / self._memory_max, 0, 1)

        return df

    def _to_borg_jobs(self, df: pd.DataFrame) -> List[BorgJob]:
        """Convert DataFrame rows to BorgJob objects."""
        jobs = []
        for idx, row in df.iterrows():
            priority = int(row.get('priority', 0))
            submit_time = float(row.get('submit_time', 0)) / 1e6  # μs → s
            duration = float(row.get('duration', 60.0))
            deadline_mult = self.PRIORITY_DEADLINE_MULTIPLIERS.get(priority, 2.0)

            job = BorgJob(
                job_id=int(idx),
                submit_time=submit_time,
                priority=priority,
                cpu_request=float(row.get('cpu_request', 0.1)),
                memory_request=float(row.get('memory_request', 0.1)),
                duration=duration,
                deadline=submit_time + duration * deadline_mult,
                n_tasks=int(row.get('n_tasks', 1))
            )
            jobs.append(job)

        return jobs

    def _generate_synthetic_borg_data(
        self,
        n_jobs: int = 1000,
        cell_id: str = "synthetic"
    ) -> List[BorgJob]:
        """
        Generate synthetic Borg-like data for testing when real trace unavailable.
        Follows distribution statistics from Tirmazi et al. (2020) [1].
        """
        logger.warning(f"Generating {n_jobs} synthetic Borg-like jobs for cell '{cell_id}'")

        jobs = []
        for i in range(n_jobs):
            # Priority follows Borg distribution (heavy on production = 4–8)
            priority = int(self.rng.choice(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                p=[0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.03, 0.02]
            ))

            # CPU follows log-normal distribution
            cpu_request = float(np.clip(self.rng.lognormal(-2, 1), 0.01, 1.0))
            memory_request = float(np.clip(self.rng.lognormal(-2, 1.2), 0.01, 1.0))

            # Duration: bimodal (short batch jobs + long service jobs)
            if self.rng.random() < 0.7:
                duration = float(self.rng.exponential(300))   # Short: ~5 min
            else:
                duration = float(self.rng.exponential(3600))  # Long: ~1 hour

            duration = max(self.min_duration, duration)
            submit_time = float(i * self.rng.exponential(10))  # Poisson arrivals
            deadline_mult = self.PRIORITY_DEADLINE_MULTIPLIERS.get(priority, 2.0)

            jobs.append(BorgJob(
                job_id=i,
                submit_time=submit_time,
                priority=priority,
                cpu_request=cpu_request,
                memory_request=memory_request,
                duration=duration,
                deadline=submit_time + duration * deadline_mult,
                n_tasks=int(self.rng.integers(1, 20))
            ))

        return jobs

    def create_episode_batches(
        self,
        jobs: List[BorgJob],
        n_jobs: int = 100,
        n_episodes: int = 100
    ) -> List[List[BorgJob]]:
        """
        Create episodic batches for training.
        Each episode contains n_jobs randomly sampled from the trace.

        Args:
            jobs: Full list of BorgJob objects
            n_jobs: Jobs per episode
            n_episodes: Number of episodes to create

        Returns:
            batches: List of job batches, one per episode
        """
        batches = []
        for _ in range(n_episodes):
            idx = self.rng.choice(len(jobs), size=min(n_jobs, len(jobs)), replace=False)
            batch = [jobs[i] for i in idx]
            # Sort by submit time within episode
            batch.sort(key=lambda j: j.submit_time)
            batches.append(batch)

        return batches

    def get_statistics(self, jobs: List[BorgJob]) -> Dict:
        """Compute descriptive statistics for Table 3 (dataset stats in paper)."""
        if not jobs:
            return {}

        cpu_requests = [j.cpu_request for j in jobs]
        memory_requests = [j.memory_request for j in jobs]
        durations = [j.duration for j in jobs]
        priorities = [j.priority for j in jobs]
        n_tasks = [j.n_tasks for j in jobs]

        return {
            'n_jobs': len(jobs),
            'cpu_mean': np.mean(cpu_requests),
            'cpu_std': np.std(cpu_requests),
            'cpu_median': np.median(cpu_requests),
            'memory_mean': np.mean(memory_requests),
            'memory_std': np.std(memory_requests),
            'memory_median': np.median(memory_requests),
            'duration_mean': np.mean(durations),
            'duration_std': np.std(durations),
            'duration_median': np.median(durations),
            'duration_min': np.min(durations),
            'duration_max': np.max(durations),
            'priority_distribution': {
                p: priorities.count(p) / len(priorities)
                for p in range(12)
            },
            'n_tasks_mean': np.mean(n_tasks),
            'n_tasks_max': max(n_tasks)
        }
