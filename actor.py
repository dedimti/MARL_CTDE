# configs/borg_config.yaml
# MARL-CTDE — Google Borg Trace 2019 Configuration

defaults:
  - default_config

# ─────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────
dataset:
  name: "GoogleBorg2019"
  path: "data/borg_trace_2019"
  cell_ids: ["a", "b", "c", "d", "e", "f", "g", "h"]  # 8 cells
  trace_duration_days: 29        # ~1 month
  preprocessing:
    normalize: true
    filter_failed_jobs: true
    min_duration_seconds: 10
    resource_columns: ["cpu_request", "memory_request", "priority"]

# ─────────────────────────────────────────────────
# EXPERIMENT SCALES
# ─────────────────────────────────────────────────
scales:
  small:
    n_nodes: 10
    n_jobs: 100
    description: "Pilot study validation"
  medium:
    n_nodes: 50
    n_jobs: 500
    description: "Medium-scale evaluation"
  large:
    n_nodes: 100
    n_jobs: 1000
    description: "Full-scale Borg simulation"

# ─────────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────
statistical:
  n_runs: 5                      # Independent runs
  confidence_level: 0.95         # CI level
  alpha_significance: 0.05       # Wilcoxon significance threshold
  effect_size: "cohen_d"

# ─────────────────────────────────────────────────
# SENSITIVITY ANALYSIS (Reward Weights)
# ─────────────────────────────────────────────────
sensitivity:
  alpha_range: [0.2, 0.3, 0.4, 0.5, 0.6]
  beta_range: [0.2, 0.3, 0.4, 0.5]
  # gamma_sla is computed as 1 - alpha - beta
