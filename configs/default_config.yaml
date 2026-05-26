# configs/default_config.yaml
# MARL-CTDE Cloud Job Scheduling — Default Configuration

# ─────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────
environment:
  name: "CloudSchedulingEnv"
  n_nodes: 10                    # Number of cluster nodes
  n_job_types: 5                 # Job priority tiers (matching Borg)
  max_queue_size: 1000           # Maximum job queue length
  time_window: 3600              # Scheduling window (seconds)
  scale: "small"                 # small / medium / large

# ─────────────────────────────────────────────────
# MARL AGENTS
# ─────────────────────────────────────────────────
agents:
  n_agents: 10                   # Number of scheduling agents
  obs_dim: 64                    # Local observation dimension
  global_state_dim: 256          # Global state dimension (critic)
  action_dim: 20                 # Action space size per agent
  hidden_dim: 256                # Hidden layer size

# ─────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────
training:
  n_episodes: 5000
  max_steps_per_episode: 200
  batch_size: 256
  buffer_size: 100000
  update_frequency: 10           # Update every N episodes
  n_epochs: 10                   # PPO epochs per update
  seed: 42

# ─────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────
optimizer:
  lr_actor: 3.0e-4
  lr_critic: 1.0e-3
  weight_decay: 1.0e-5
  grad_clip: 0.5

# ─────────────────────────────────────────────────
# PPO HYPERPARAMETERS
# ─────────────────────────────────────────────────
ppo:
  gamma: 0.99                    # Discount factor
  gae_lambda: 0.95               # GAE lambda
  clip_epsilon: 0.2              # PPO clipping
  entropy_coef: 0.01             # Entropy regularization
  value_loss_coef: 0.5           # Value loss coefficient

# ─────────────────────────────────────────────────
# MULTI-OBJECTIVE REWARD WEIGHTS
# ─────────────────────────────────────────────────
reward:
  alpha: 0.4                     # Makespan weight
  beta: 0.3                      # Energy consumption weight
  gamma_sla: 0.3                 # SLA violation weight
  # Note: alpha + beta + gamma_sla = 1.0

# ─────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────
logging:
  log_dir: "results/logs"
  checkpoint_dir: "results/checkpoints"
  eval_frequency: 100            # Evaluate every N episodes
  save_frequency: 500            # Save checkpoint every N episodes
  tensorboard: true
