"""
src/agents/marl_agent.py
Main MARL-CTDE Agent — MAPPO-based Training and Execution

Implements Cooperative MARL with:
    - Centralized Training (shared critic with global state)
    - Decentralized Execution (each agent uses only local observation)
    - Multi-objective reward optimization

References:
    [6]  Hady et al. (2025) - MARL resources allocation survey
    [21] Shi et al. (2025) - CTPDE: Policy distillation
    [22] Wang et al. (2024) - Coordination as inference
    [25] Ma et al. (2025)  - MAPPO with priority-gated attention
    [26] Lao et al. (2024) - IMAPPO for resource allocation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from loguru import logger

from .actor import ActorNetwork
from .critic import SharedCriticNetwork


class ReplayBuffer:
    """
    Experience replay buffer for MARL.
    Stores transitions for batch updates.
    """

    def __init__(self, buffer_size: int, n_agents: int):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.buffer = deque(maxlen=buffer_size)

    def add(
        self,
        obs: np.ndarray,           # (n_agents, obs_dim)
        global_state: np.ndarray,  # (global_state_dim,)
        actions: np.ndarray,       # (n_agents,)
        log_probs: np.ndarray,     # (n_agents,)
        rewards: np.ndarray,       # (n_agents,)  -- multi-objective combined
        next_obs: np.ndarray,
        next_global_state: np.ndarray,
        dones: np.ndarray          # (n_agents,)
    ):
        self.buffer.append({
            'obs': obs,
            'global_state': global_state,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'next_obs': next_obs,
            'next_global_state': next_global_state,
            'dones': dones
        })

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        return {
            'obs': torch.FloatTensor(np.stack([b['obs'] for b in batch])),
            'global_state': torch.FloatTensor(np.stack([b['global_state'] for b in batch])),
            'actions': torch.LongTensor(np.stack([b['actions'] for b in batch])),
            'log_probs': torch.FloatTensor(np.stack([b['log_probs'] for b in batch])),
            'rewards': torch.FloatTensor(np.stack([b['rewards'] for b in batch])),
            'next_obs': torch.FloatTensor(np.stack([b['next_obs'] for b in batch])),
            'next_global_state': torch.FloatTensor(np.stack([b['next_global_state'] for b in batch])),
            'dones': torch.FloatTensor(np.stack([b['dones'] for b in batch]))
        }

    def __len__(self):
        return len(self.buffer)


class MARLCTDEAgent:
    """
    Main MARL-CTDE Agent.

    Manages n_agents actor networks + 1 shared critic.
    Implements MAPPO (Multi-Agent PPO) training.

    Training loop:
        1. Collect trajectories using all actors
        2. Compute advantages using shared critic
        3. Update all actors via PPO clipping loss
        4. Update shared critic via MSE value loss

    Execution loop:
        1. Each actor independently observes local state
        2. Each actor selects action (no critic needed)
        3. Actions are executed in cloud environment
    """

    def __init__(self, config: dict, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.config = config
        self.device = device
        self.n_agents = config['agents']['n_agents']

        # Initialize n_agents actor networks (decentralized)
        self.actors = nn.ModuleList([
            ActorNetwork(
                obs_dim=config['agents']['obs_dim'],
                action_dim=config['agents']['action_dim'],
                hidden_dim=config['agents']['hidden_dim']
            ) for _ in range(self.n_agents)
        ]).to(device)

        # Initialize 1 shared critic (centralized)
        self.critic = SharedCriticNetwork(
            global_state_dim=config['agents']['global_state_dim'],
            n_agents=self.n_agents,
            hidden_dim=config['agents']['hidden_dim']
        ).to(device)

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(
                actor.parameters(),
                lr=config['optimizer']['lr_actor'],
                weight_decay=config['optimizer']['weight_decay']
            ) for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config['optimizer']['lr_critic'],
            weight_decay=config['optimizer']['weight_decay']
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            buffer_size=config['training']['buffer_size'],
            n_agents=self.n_agents
        )

        # PPO parameters
        self.gamma = config['ppo']['gamma']
        self.gae_lambda = config['ppo']['gae_lambda']
        self.clip_epsilon = config['ppo']['clip_epsilon']
        self.entropy_coef = config['ppo']['entropy_coef']
        self.value_loss_coef = config['ppo']['value_loss_coef']
        self.grad_clip = config['optimizer']['grad_clip']
        self.n_epochs = config['training']['n_epochs']
        self.batch_size = config['training']['batch_size']

        # Training statistics
        self.train_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'entropy': [],
            'episode_rewards': []
        }

        logger.info(f"MARL-CTDE initialized: {self.n_agents} agents on {device}")

    def select_actions(
        self,
        observations: List[np.ndarray],
        action_masks: Optional[List[np.ndarray]] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions for all agents (DECENTRALIZED EXECUTION).
        Each agent uses only its own local observation.

        Args:
            observations: List of local observations, one per agent
            action_masks: List of valid action masks, one per agent
            deterministic: Use greedy policy (for evaluation)

        Returns:
            actions: Selected actions for all agents (n_agents,)
            log_probs: Log probabilities of selected actions (n_agents,)
        """
        actions = []
        log_probs = []

        for i, actor in enumerate(self.actors):
            obs_tensor = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            mask = None
            if action_masks is not None:
                mask = torch.BoolTensor(action_masks[i]).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, _ = actor.get_action(obs_tensor, mask, deterministic)

            actions.append(action.cpu().numpy()[0])
            log_probs.append(log_prob.cpu().numpy()[0])

        return np.array(actions), np.array(log_probs)

    def update(self) -> Dict[str, float]:
        """
        PPO update step (CENTRALIZED TRAINING).
        Updates all actors and the shared critic.

        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {}

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for epoch in range(self.n_epochs):
            batch = self.buffer.sample(self.batch_size)

            # Move batch to device
            obs = batch['obs'].to(self.device)              # (batch, n_agents, obs_dim)
            global_state = batch['global_state'].to(self.device)   # (batch, global_state_dim)
            actions = batch['actions'].to(self.device)      # (batch, n_agents)
            old_log_probs = batch['log_probs'].to(self.device)     # (batch, n_agents)
            rewards = batch['rewards'].to(self.device)      # (batch, n_agents)
            next_global_state = batch['next_global_state'].to(self.device)
            dones = batch['dones'].to(self.device)

            # ── CRITIC UPDATE ──────────────────────────────────────
            # Compute values and advantages
            with torch.no_grad():
                next_values = self.critic(next_global_state).squeeze(-1)
                next_values = next_values.unsqueeze(-1).expand(-1, self.n_agents)

            values = self.critic(global_state).squeeze(-1)
            values = values.unsqueeze(-1).expand(-1, self.n_agents)

            # GAE advantages
            mean_rewards = rewards.mean(dim=-1, keepdim=True).expand(-1, self.n_agents)
            advantages = mean_rewards + self.gamma * next_values * (1 - dones) - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + values.detach()

            # Critic loss (MSE)
            critic_values = self.critic(global_state).squeeze(-1)
            critic_values = critic_values.unsqueeze(-1).expand(-1, self.n_agents)
            critic_loss = self.value_loss_coef * F.mse_loss(critic_values, returns.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()

            # ── ACTOR UPDATES ─────────────────────────────────────
            actor_loss_total = 0
            entropy_total = 0

            for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
                agent_obs = obs[:, i, :]           # (batch, obs_dim)
                agent_actions = actions[:, i]       # (batch,)
                agent_old_log_probs = old_log_probs[:, i]  # (batch,)
                agent_advantages = advantages[:, i].detach()  # (batch,)

                # Evaluate current policy on collected actions
                new_log_probs, entropy = actor.evaluate_actions(agent_obs, agent_actions)

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - agent_old_log_probs)
                surr1 = ratio * agent_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * agent_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy regularization (encourages exploration)
                entropy_loss = -self.entropy_coef * entropy.mean()
                total_loss = actor_loss + entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), self.grad_clip)
                optimizer.step()

                actor_loss_total += actor_loss.item()
                entropy_total += entropy.mean().item()

            total_actor_loss += actor_loss_total / self.n_agents
            total_critic_loss += critic_loss.item()
            total_entropy += entropy_total / self.n_agents

        metrics = {
            'actor_loss': total_actor_loss / self.n_epochs,
            'critic_loss': total_critic_loss / self.n_epochs,
            'entropy': total_entropy / self.n_epochs
        }

        # Update training stats
        self.train_stats['actor_losses'].append(metrics['actor_loss'])
        self.train_stats['critic_losses'].append(metrics['critic_loss'])
        self.train_stats['entropy'].append(metrics['entropy'])

        return metrics

    def save(self, path: str):
        """Save all model weights."""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'config': self.config,
            'train_stats': self.train_stats
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load(self, path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])
        self.train_stats = checkpoint.get('train_stats', self.train_stats)
        logger.info(f"Checkpoint loaded from {path}")


# Alias for importing
import torch.nn.functional as F
