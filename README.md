"""
src/agents/critic.py
Centralized Shared Critic Network for MARL-CTDE Cloud Job Scheduling

The shared critic has access to the GLOBAL state during training only.
It provides value estimates to guide all agents' policy updates.
During execution, the critic is NOT used — only actors run.

This implements the CTDE paradigm:
    - Centralized Training: critic uses global state
    - Decentralized Execution: only actors are deployed

References:
    [21] Shi et al. (2025) - CTPDE framework
    [22] Wang et al. (2024) - Coordination as inference
    [23] Liu et al. (2024) - Cournot Policy Model MARL
    [24] Zhang et al. (2024) - QDAP cooperative MARL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SharedCriticNetwork(nn.Module):
    """
    Centralized Shared Critic Network.

    Architecture:
        - Input: global state (all agents' observations + cluster resource state)
        - Cross-Agent Attention: model inter-agent dependencies
        - MLP: value estimation
        - Output: scalar value V(s) for advantage computation

    Key design decisions:
        1. Shared across all agents (reduces parameters, improves coordination)
        2. Cross-attention models agent interaction patterns
        3. Global state includes full cluster resource utilization

    Only used during TRAINING. Not deployed during execution.
    """

    def __init__(
        self,
        global_state_dim: int,
        n_agents: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(SharedCriticNetwork, self).__init__()

        self.global_state_dim = global_state_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Global state embedding
        self.state_embed = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Cross-agent attention: model coordination between agents
        self.cross_agent_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)

        # Value MLP
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        global_state: torch.Tensor,
        agent_obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute state value V(s).

        Args:
            global_state: Full cluster state (batch_size, global_state_dim)
            agent_obs: All agents' observations stacked (batch_size, n_agents, obs_dim)
                       Used for cross-agent attention if provided

        Returns:
            value: Estimated state value (batch_size, 1)
        """
        # Embed global state
        state_emb = self.state_embed(global_state)  # (batch, hidden_dim)

        if agent_obs is not None:
            # Cross-agent attention over all agents' observations
            # Project agent obs to hidden dim
            batch_size = agent_obs.shape[0]

            # Use state embedding as query, agent observations as key/value
            state_emb_expanded = state_emb.unsqueeze(1)  # (batch, 1, hidden_dim)

            # Simple projection of agent observations
            agent_emb = agent_obs.mean(dim=-1, keepdim=True)
            agent_emb = agent_emb.expand(-1, -1, self.hidden_dim)

            attended, _ = self.cross_agent_attention(
                state_emb_expanded,  # query
                agent_emb,           # key
                agent_emb            # value
            )
            state_emb = self.attention_norm(state_emb + attended.squeeze(1))

        # Compute value
        value = self.value_net(state_emb)

        return value

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward signals (batch_size, T)
            values: Estimated values V(s_t) (batch_size, T)
            next_values: Estimated values V(s_{t+1}) (batch_size, T)
            dones: Episode termination flags (batch_size, T)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: GAE advantages (batch_size, T)
        """
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(rewards.shape[-1])):
            gae = deltas[..., t] + gamma * gae_lambda * (1 - dones[..., t]) * gae
            advantages[..., t] = gae

        return advantages
