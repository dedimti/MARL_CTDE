"""
src/environment/reward.py
Multi-Objective Reward Function for MARL-CTDE Cloud Job Scheduling

R = α·(-makespan_normalized) + β·(-energy_normalized) + γ·(-SLA_violation_rate)

References:
    [12] Fayaz et al. (2025) - RL-MOTS multi-objective cloud-edge
    [13] Khan et al. (2025)  - Dynamic multi-objective RL cloud
    [29] Chandrasiri (2025)  - Energy-efficient GNN+PPO workflow
    [31] Moazeni et al. (2023) - Multi-objective teaching-learning cloud
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SchedulingMetrics:
    """Container for raw scheduling metrics before reward computation."""
    makespan: float          # Total job completion time (seconds)
    energy_consumed: float   # Total energy consumed (Joules)
    sla_violations: int      # Number of jobs exceeding deadline
    total_jobs: int          # Total jobs scheduled
    resource_utilization: float  # Mean CPU/memory utilization (0–1)
    fairness_index: float    # Jain's Fairness Index (0–1)


class MultiObjectiveReward:
    """
    Multi-objective reward function combining:
        1. Makespan minimization
        2. Energy consumption minimization
        3. SLA violation rate minimization

    Reward:
        R = α·(-makespan_norm) + β·(-energy_norm) + γ·(-sla_violation_rate)

    Constraint: α + β + γ = 1.0

    Normalization:
        - Makespan: normalized by baseline heuristic makespan
        - Energy: normalized by idle power × time
        - SLA violation: rate = violations / total_jobs
    """

    def __init__(
        self,
        alpha: float = 0.4,     # Makespan weight
        beta: float = 0.3,      # Energy weight
        gamma_sla: float = 0.3, # SLA violation weight
        baseline_makespan: float = 1.0,
        baseline_energy: float = 1.0,
        penalty_scale: float = 10.0
    ):
        assert abs(alpha + beta + gamma_sla - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {alpha + beta + gamma_sla:.4f}"

        self.alpha = alpha
        self.beta = beta
        self.gamma_sla = gamma_sla
        self.baseline_makespan = baseline_makespan
        self.baseline_energy = baseline_energy
        self.penalty_scale = penalty_scale

        # Running statistics for adaptive normalization
        self.makespan_history = []
        self.energy_history = []

    def compute(
        self,
        metrics: SchedulingMetrics,
        step: int = 0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute multi-objective reward.

        Args:
            metrics: Raw scheduling metrics for current step
            step: Current training step (for adaptive normalization)

        Returns:
            reward: Scalar reward value
            breakdown: Dictionary of individual reward components
        """
        # ── Normalize Makespan ─────────────────────────────────
        if self.baseline_makespan > 0:
            makespan_norm = metrics.makespan / self.baseline_makespan
        else:
            makespan_norm = 1.0

        # Clip to reasonable range
        makespan_norm = np.clip(makespan_norm, 0.0, 5.0)
        makespan_reward = -makespan_norm

        # ── Normalize Energy ───────────────────────────────────
        if self.baseline_energy > 0:
            energy_norm = metrics.energy_consumed / self.baseline_energy
        else:
            energy_norm = 1.0

        energy_norm = np.clip(energy_norm, 0.0, 5.0)
        energy_reward = -energy_norm

        # ── SLA Violation Rate ─────────────────────────────────
        if metrics.total_jobs > 0:
            sla_violation_rate = metrics.sla_violations / metrics.total_jobs
        else:
            sla_violation_rate = 0.0

        sla_violation_rate = np.clip(sla_violation_rate, 0.0, 1.0)
        sla_reward = -sla_violation_rate

        # ── Combined Multi-Objective Reward ────────────────────
        reward = (
            self.alpha * makespan_reward +
            self.beta * energy_reward +
            self.gamma_sla * sla_reward
        )

        # ── Bonus for high resource utilization ───────────────
        utilization_bonus = 0.1 * metrics.resource_utilization

        # ── Bonus for fairness ────────────────────────────────
        fairness_bonus = 0.05 * metrics.fairness_index

        total_reward = reward + utilization_bonus + fairness_bonus

        # Breakdown for logging
        breakdown = {
            'total_reward': total_reward,
            'makespan_reward': self.alpha * makespan_reward,
            'energy_reward': self.beta * energy_reward,
            'sla_reward': self.gamma_sla * sla_reward,
            'utilization_bonus': utilization_bonus,
            'fairness_bonus': fairness_bonus,
            'makespan_norm': makespan_norm,
            'energy_norm': energy_norm,
            'sla_violation_rate': sla_violation_rate
        }

        return total_reward, breakdown

    def update_baselines(
        self,
        makespan: float,
        energy: float
    ):
        """
        Update normalization baselines using running average.
        Called after each episode to adapt to workload characteristics.
        """
        self.makespan_history.append(makespan)
        self.energy_history.append(energy)

        # Use running average of last 100 episodes
        window = 100
        if len(self.makespan_history) >= 5:
            self.baseline_makespan = np.mean(self.makespan_history[-window:])
        if len(self.energy_history) >= 5:
            self.baseline_energy = np.mean(self.energy_history[-window:])

    @staticmethod
    def compute_jains_fairness(allocations: np.ndarray) -> float:
        """
        Compute Jain's Fairness Index.
        JFI = (Σxᵢ)² / (n · Σxᵢ²)

        Args:
            allocations: Resource allocations per job/agent

        Returns:
            fairness: Jain's Fairness Index (0=unfair, 1=perfectly fair)
        """
        if len(allocations) == 0 or np.sum(allocations ** 2) == 0:
            return 1.0

        n = len(allocations)
        numerator = np.sum(allocations) ** 2
        denominator = n * np.sum(allocations ** 2)

        return float(numerator / denominator)

    def sensitivity_analysis(
        self,
        metrics: SchedulingMetrics,
        alpha_range: list = [0.2, 0.3, 0.4, 0.5, 0.6],
        beta_range: list = [0.2, 0.3, 0.4, 0.5]
    ) -> Dict:
        """
        Run reward weight sensitivity analysis.
        Computes reward for all valid (alpha, beta, gamma) combinations.

        Returns:
            results: Dict mapping (alpha, beta, gamma) -> reward
        """
        results = {}
        for alpha in alpha_range:
            for beta in beta_range:
                gamma = 1.0 - alpha - beta
                if gamma < 0.1:  # Minimum SLA weight
                    continue
                temp_reward = MultiObjectiveReward(alpha, beta, gamma,
                                                   self.baseline_makespan,
                                                   self.baseline_energy)
                reward, _ = temp_reward.compute(metrics)
                results[(round(alpha, 2), round(beta, 2), round(gamma, 2))] = reward

        return results
