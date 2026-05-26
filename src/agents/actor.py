"""
src/agents/actor.py
Decentralized Actor Network for MARL-CTDE Cloud Job Scheduling

Each agent has its own actor that makes decisions based on LOCAL observations only.
This enables decentralized execution in production (no inter-agent communication needed).

Reference:
    [21] Shi et al. (2025) - CTPDE: Policy distillation for decentralized execution
    [22] Wang et al. (2024) - Coordination as inference in MARL
    [25] Ma et al. (2025)  - MAPPO with priority-gated attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional


class AttentionLayer(nn.Module):
    """
    Self-attention mechanism for processing local job queue observations.
    Allows agent to focus on the most relevant jobs in its local queue.
    """

    def __init__(self, input_dim: int, num_heads: int = 4):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)


class ActorNetwork(nn.Module):
    """
    Decentralized Actor Network.

    Architecture:
        - Input: local observation (resource state + job queue)
        - Attention: focus on relevant jobs
        - MLP: policy computation
        - Output: action probability distribution

    Used during BOTH training and execution.
    During execution: only uses local observation (no global state needed).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(ActorNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Attention for job queue processing
        self.attention = AttentionLayer(hidden_dim, num_heads)

        # Policy MLP
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable training."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Local observation tensor (batch_size, obs_dim)
            action_mask: Boolean mask for invalid actions (batch_size, action_dim)

        Returns:
            action_probs: Action probability distribution
            log_probs: Log probabilities
        """
        # Input embedding
        x = self.input_embed(obs)

        # Reshape for attention (treat obs as sequence of length 1 for simplicity)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        x = self.attention(x)
        x = x.squeeze(1)    # (batch, hidden_dim)

        # Compute logits
        logits = self.policy_net(x)

        # Apply action mask (mask invalid actions with -inf)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        # Softmax to get probabilities
        action_probs = F.softmax(logits, dim=-1)

        # Add small epsilon for numerical stability
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        log_probs = torch.log(action_probs)

        return action_probs, log_probs

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Local observation
            action_mask: Valid action mask
            deterministic: If True, take argmax (for evaluation)

        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            entropy: Policy entropy (for regularization)
        """
        action_probs, log_probs = self.forward(obs, action_mask)

        dist = Categorical(probs=action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()

        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, action_log_prob, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.
        Used during PPO update step.

        Args:
            obs: Local observations (batch)
            actions: Actions taken (batch)
            action_mask: Valid action masks (batch)

        Returns:
            log_probs: Log probabilities of taken actions
            entropy: Policy entropy
        """
        action_probs, _ = self.forward(obs, action_mask)
        dist = Categorical(probs=action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy
