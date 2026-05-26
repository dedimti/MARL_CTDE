"""
src/environment/cloud_env.py
Cloud Scheduling Environment — Gymnasium-compatible

Simulates a heterogeneous cloud cluster for MARL-based job scheduling.
Each agent manages a subset of nodes and schedules incoming jobs.

References:
    [1]  Tirmazi et al. (2020) - Borg: the Next Generation
    [36] Buyya et al. (2009)  - CloudSim toolkit
    [37] Ahmad et al. (2025)  - CBA-HDL workload forecasting
    [45] Liu et al. (2025)    - MARL edge resource orchestration
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from loguru import logger

from .borg_loader import BorgJob
from .reward import MultiObjectiveReward, SchedulingMetrics


@dataclass
class CloudNode:
    """Represents a single compute node in the cluster."""
    node_id: int
    cpu_capacity: float        # Total CPU (normalized 0–1)
    memory_capacity: float     # Total memory (normalized 0–1)
    cpu_used: float = 0.0
    memory_used: float = 0.0
    power_idle: float = 0.1    # Idle power consumption (normalized)
    power_max: float = 1.0     # Max power consumption (normalized)
    assigned_jobs: List = field(default_factory=list)

    @property
    def cpu_available(self) -> float:
        return max(0.0, self.cpu_capacity - self.cpu_used)

    @property
    def memory_available(self) -> float:
        return max(0.0, self.memory_capacity - self.memory_used)

    @property
    def utilization(self) -> float:
        return (self.cpu_used / self.cpu_capacity +
                self.memory_used / self.memory_capacity) / 2.0

    @property
    def current_power(self) -> float:
        """Linear power model: P = P_idle + (P_max - P_idle) * utilization"""
        return self.power_idle + (self.power_max - self.power_idle) * self.utilization

    def can_accommodate(self, job: BorgJob) -> bool:
        return (self.cpu_available >= job.cpu_request and
                self.memory_available >= job.memory_request)

    def assign_job(self, job: BorgJob):
        self.cpu_used += job.cpu_request
        self.memory_used += job.memory_request
        self.assigned_jobs.append(job.job_id)

    def release_job(self, job: BorgJob):
        self.cpu_used = max(0.0, self.cpu_used - job.cpu_request)
        self.memory_used = max(0.0, self.memory_used - job.memory_request)
        if job.job_id in self.assigned_jobs:
            self.assigned_jobs.remove(job.job_id)


class CloudSchedulingEnv(gym.Env):
    """
    Multi-agent Cloud Job Scheduling Environment.

    Observation (per agent):
        - Node resource state (cpu_available, memory_available, utilization)
        - Job queue features (priority, cpu_request, memory_request, urgency)
        - Time features (current_time, queue_length)

    Action (per agent):
        - Assign top-queue job to one of K nodes (0 = defer)

    Reward:
        Multi-objective: makespan + energy + SLA violation

    Episode:
        - Terminates when all jobs in batch are scheduled or max_steps reached
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.n_agents = config['agents']['n_agents']
        self.n_nodes = config['environment']['n_nodes']
        self.max_queue_size = config['environment']['max_queue_size']
        self.max_steps = config['training']['max_steps_per_episode']

        # Observation and action dimensions
        self.obs_dim = config['agents']['obs_dim']
        self.action_dim = config['agents']['action_dim']
        self.global_state_dim = config['agents']['global_state_dim']

        # Observation space per agent
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )

        # Action space per agent (assign to node 0..n_nodes-1, or defer=-1)
        self.action_space = spaces.Discrete(self.n_nodes + 1)

        # Reward function
        self.reward_fn = MultiObjectiveReward(
            alpha=config['reward']['alpha'],
            beta=config['reward']['beta'],
            gamma_sla=config['reward']['gamma_sla']
        )

        # Environment state
        self.nodes: List[CloudNode] = []
        self.job_queue: List[BorgJob] = []
        self.current_time: float = 0.0
        self.step_count: int = 0
        self.completed_jobs: List[BorgJob] = []
        self.sla_violations: int = 0
        self.total_energy: float = 0.0
        self.episode_jobs: List[BorgJob] = []

        # Per-step metrics
        self._step_metrics = []

        self._initialize_nodes()
        logger.info(f"CloudSchedulingEnv: {self.n_nodes} nodes, "
                   f"{self.n_agents} agents, obs_dim={self.obs_dim}")

    def _initialize_nodes(self):
        """Initialize heterogeneous cluster nodes."""
        self.nodes = []
        rng = np.random.default_rng(42)

        for i in range(self.n_nodes):
            # Heterogeneous capacity: mix of small, medium, large nodes
            node_type = i % 3
            if node_type == 0:   # Small node
                cpu = rng.uniform(0.3, 0.5)
                mem = rng.uniform(0.3, 0.5)
            elif node_type == 1:  # Medium node
                cpu = rng.uniform(0.5, 0.8)
                mem = rng.uniform(0.5, 0.8)
            else:                 # Large node
                cpu = rng.uniform(0.8, 1.0)
                mem = rng.uniform(0.8, 1.0)

            self.nodes.append(CloudNode(
                node_id=i,
                cpu_capacity=cpu,
                memory_capacity=mem,
                power_idle=0.05 + 0.1 * node_type,
                power_max=0.5 + 0.3 * node_type
            ))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[List[np.ndarray], Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset nodes
        for node in self.nodes:
            node.cpu_used = 0.0
            node.memory_used = 0.0
            node.assigned_jobs = []

        # Load new job batch
        if options and 'jobs' in options:
            self.episode_jobs = list(options['jobs'])
        else:
            self.episode_jobs = []

        self.job_queue = list(self.episode_jobs[:self.max_queue_size])
        self.current_time = 0.0
        self.step_count = 0
        self.completed_jobs = []
        self.sla_violations = 0
        self.total_energy = 0.0
        self._step_metrics = []

        observations = self._get_all_observations()
        global_state = self._get_global_state()

        info = {
            'n_jobs': len(self.episode_jobs),
            'global_state': global_state
        }

        return observations, info

    def step(
        self,
        actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """
        Execute one scheduling step.

        Args:
            actions: List of actions from all agents (n_agents,)
                     action[i] ∈ {0..n_nodes}: assign top job to node i
                     action[i] = n_nodes: defer (no assignment)

        Returns:
            observations: New observations for all agents
            rewards: Rewards for all agents (same reward = cooperative)
            terminated: Episode ended (all jobs scheduled)
            truncated: Episode ended (max steps reached)
            info: Additional info including metrics
        """
        self.step_count += 1

        # Track energy consumed this step
        step_duration = 1.0  # 1 second per step
        step_energy = sum(node.current_power * step_duration for node in self.nodes)
        self.total_energy += step_energy

        # Process actions from all agents
        assignments_made = 0
        for agent_id, action in enumerate(actions):
            if not self.job_queue:
                break

            # Get the job this agent is responsible for
            agent_job_idx = agent_id % len(self.job_queue) if self.job_queue else 0
            if agent_job_idx >= len(self.job_queue):
                continue

            job = self.job_queue[agent_job_idx]

            # Action = node_id to assign to (n_nodes = defer)
            if action < self.n_nodes:
                target_node = self.nodes[action]
                if target_node.can_accommodate(job):
                    target_node.assign_job(job)
                    self.job_queue.pop(agent_job_idx)
                    self.completed_jobs.append(job)
                    assignments_made += 1

                    # Check SLA
                    finish_time = self.current_time + job.duration
                    if finish_time > job.deadline:
                        self.sla_violations += 1

        # Advance time
        self.current_time += step_duration

        # Release completed jobs from nodes
        self._release_completed_jobs()

        # Compute metrics
        metrics = self._compute_step_metrics()
        reward, reward_breakdown = self.reward_fn.compute(metrics)

        # Same cooperative reward for all agents
        rewards = [reward] * self.n_agents

        # Check termination
        terminated = len(self.job_queue) == 0
        truncated = self.step_count >= self.max_steps

        # Get new observations
        observations = self._get_all_observations()
        global_state = self._get_global_state()

        info = {
            'metrics': metrics,
            'reward_breakdown': reward_breakdown,
            'assignments_made': assignments_made,
            'queue_length': len(self.job_queue),
            'completed_jobs': len(self.completed_jobs),
            'sla_violations': self.sla_violations,
            'total_energy': self.total_energy,
            'global_state': global_state
        }

        return observations, rewards, terminated, truncated, info

    def _release_completed_jobs(self):
        """Release resources from jobs that have finished executing."""
        for node in self.nodes:
            jobs_to_release = []
            for job in self.episode_jobs:
                if (job.job_id in node.assigned_jobs and
                        self.current_time >= job.submit_time + job.duration):
                    jobs_to_release.append(job)

            for job in jobs_to_release:
                node.release_job(job)

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """
        Get local observation for agent_id.
        Agent observes its assigned nodes + top jobs in queue.
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0

        # Node features (each agent monitors n_nodes/n_agents nodes)
        nodes_per_agent = max(1, self.n_nodes // self.n_agents)
        start_node = agent_id * nodes_per_agent
        end_node = min(start_node + nodes_per_agent, self.n_nodes)

        for node_i in range(start_node, end_node):
            if idx + 4 >= self.obs_dim:
                break
            node = self.nodes[node_i]
            obs[idx] = node.cpu_available
            obs[idx + 1] = node.memory_available
            obs[idx + 2] = node.utilization
            obs[idx + 3] = node.current_power
            idx += 4

        # Job queue features (top 5 jobs)
        for job_i in range(min(5, len(self.job_queue))):
            if idx + 5 >= self.obs_dim:
                break
            job = self.job_queue[job_i]
            obs[idx] = job.priority / 11.0           # Normalized priority
            obs[idx + 1] = job.cpu_request
            obs[idx + 2] = job.memory_request
            obs[idx + 3] = job.duration / 3600.0     # Normalized duration
            urgency = max(0, (job.deadline - self.current_time) / job.duration)
            obs[idx + 4] = min(urgency, 5.0) / 5.0  # Normalized urgency
            idx += 5

        # Global queue info
        if idx < self.obs_dim:
            obs[idx] = len(self.job_queue) / self.max_queue_size
        if idx + 1 < self.obs_dim:
            obs[idx + 1] = self.step_count / self.max_steps

        return obs

    def _get_all_observations(self) -> List[np.ndarray]:
        return [self._get_observation(i) for i in range(self.n_agents)]

    def _get_global_state(self) -> np.ndarray:
        """Get global state for centralized critic (training only)."""
        state = np.zeros(self.global_state_dim, dtype=np.float32)
        idx = 0

        # All node states
        for node in self.nodes:
            if idx + 3 >= self.global_state_dim:
                break
            state[idx] = node.cpu_available
            state[idx + 1] = node.memory_available
            state[idx + 2] = node.utilization
            idx += 3

        # Queue summary
        if self.job_queue:
            if idx < self.global_state_dim:
                state[idx] = len(self.job_queue) / self.max_queue_size
            if idx + 1 < self.global_state_dim:
                state[idx + 1] = np.mean([j.priority for j in self.job_queue]) / 11.0
            if idx + 2 < self.global_state_dim:
                state[idx + 2] = self.sla_violations / max(1, len(self.episode_jobs))

        return state

    def _compute_step_metrics(self) -> SchedulingMetrics:
        """Compute metrics for current step."""
        # Makespan: time to complete all assigned jobs
        if self.completed_jobs:
            makespan = self.current_time + np.mean([j.duration for j in self.job_queue]) \
                       if self.job_queue else self.current_time
        else:
            makespan = self.current_time

        # Resource allocations for Jain's fairness
        allocations = np.array([j.cpu_request for j in self.completed_jobs]) \
                      if self.completed_jobs else np.array([1.0])

        return SchedulingMetrics(
            makespan=makespan,
            energy_consumed=self.total_energy,
            sla_violations=self.sla_violations,
            total_jobs=len(self.episode_jobs),
            resource_utilization=np.mean([n.utilization for n in self.nodes]),
            fairness_index=MultiObjectiveReward.compute_jains_fairness(allocations)
        )

    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """Get valid action mask for agent."""
        mask = np.zeros(self.n_nodes + 1, dtype=bool)
        mask[-1] = True  # Defer always valid

        if not self.job_queue:
            return mask

        agent_job_idx = agent_id % len(self.job_queue)
        if agent_job_idx < len(self.job_queue):
            job = self.job_queue[agent_job_idx]
            for i, node in enumerate(self.nodes):
                if node.can_accommodate(job):
                    mask[i] = True

        return mask
