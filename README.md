# MARL-CTDE Cloud Job Scheduling
## Cooperative Multi-Agent Reinforcement Learning with Centralized Training and Decentralized Execution for Distributed Cloud Job Scheduling

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Journal](https://img.shields.io/badge/Target-FGCS%20Scopus%20Q1-red)](https://www.sciencedirect.com/journal/future-generation-computer-systems)

---

## 📋 Overview

This repository contains the implementation of **MARL-CTDE**, a cooperative multi-agent reinforcement learning framework for distributed cloud job scheduling, evaluated on the **Google Borg Workload Traces (2019)**.

> **Paper:** "Cooperative Multi-Agent Reinforcement Learning with Centralized Training and Decentralized Execution for Distributed Cloud Job Scheduling: An Empirical Study on Google Borg Workload Traces"
> **Journal:** Future Generation Computer Systems (Elsevier, Scopus Q1)
> **Status:** Under Review

---

## 🏗️ Architecture

```
MARL-CTDE Framework
├── Centralized Training
│   └── Shared Critic Network (global state access)
├── Decentralized Execution
│   └── Per-Agent Actor Network (local observation only)
└── Multi-Objective Reward
    └── R = α·makespan + β·energy + γ·SLA_violation
```

---

## 📁 Repository Structure

```
marl_cloud_scheduling/
├── README.md
├── requirements.txt
├── configs/
│   ├── default_config.yaml        # Default hyperparameters
│   └── borg_config.yaml           # Google Borg-specific config
├── src/
│   ├── agents/
│   │   ├── actor.py               # Decentralized actor network
│   │   ├── critic.py              # Centralized shared critic
│   │   └── marl_agent.py          # Main MARL agent (MAPPO-based)
│   ├── environment/
│   │   ├── cloud_env.py           # Cloud scheduling environment
│   │   ├── borg_loader.py         # Google Borg trace loader
│   │   └── reward.py              # Multi-objective reward function
│   └── utils/
│       ├── metrics.py             # Evaluation metrics
│       ├── logger.py              # Experiment logging
│       └── statistical.py        # Wilcoxon, CI, Cohen's d
├── scripts/
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── baseline.py                # Baseline algorithms
├── tests/
│   └── test_environment.py        # Unit tests
└── results/
    └── placeholder/               # Results go here after running
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/[YOUR_USERNAME]/marl-cloud-scheduling.git
cd marl-cloud-scheduling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset — Google Borg Trace 2019

Download the Google Borg trace from:
🔗 https://github.com/google/cluster-data

```bash
# After downloading, place in:
mkdir -p data/borg_trace_2019
# Copy trace files to data/borg_trace_2019/
```

Expected structure:
```
data/borg_trace_2019/
├── job_events/
├── task_events/
├── machine_events/
└── machine_attributes/
```

---

## 🚀 Training

```bash
# Train MARL-CTDE (default config)
python scripts/train.py --config configs/borg_config.yaml

# Train with custom hyperparameters
python scripts/train.py \
    --n_agents 10 \
    --n_episodes 5000 \
    --lr_actor 3e-4 \
    --lr_critic 1e-3 \
    --alpha 0.4 \
    --beta 0.3 \
    --gamma_sla 0.3

# Train baselines for comparison
python scripts/baseline.py --algorithm dqn
python scripts/baseline.py --algorithm ppo
python scripts/baseline.py --algorithm round_robin
python scripts/baseline.py --algorithm min_min
python scripts/baseline.py --algorithm ippo
```

---

## 📈 Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint results/checkpoints/best_model.pt \
    --scale small    # small / medium / large
    --n_runs 5       # for statistical significance

# Generate all result tables
python scripts/evaluate.py --generate_tables
```

---

## 🔬 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_agents` | 10 | Number of scheduling agents |
| `lr_actor` | 3e-4 | Actor learning rate |
| `lr_critic` | 1e-3 | Critic learning rate |
| `gamma` | 0.99 | Discount factor |
| `clip_epsilon` | 0.2 | PPO clipping parameter |
| `batch_size` | 256 | Mini-batch size |
| `buffer_size` | 100000 | Experience replay buffer size |
| `alpha` (makespan) | 0.4 | Reward weight — makespan |
| `beta` (energy) | 0.3 | Reward weight — energy |
| `gamma_sla` (SLA) | 0.3 | Reward weight — SLA violation |
| `n_episodes` | 5000 | Training episodes |
| `seed` | 42 | Random seed (reproducibility) |

---

## 📋 Evaluation Metrics

| Metric | Formula | Reference |
|--------|---------|-----------|
| Makespan | Total completion time | Lower = better |
| Energy Consumption | Σ power × time | Lower = better |
| SLA Violation Rate | % jobs exceeding deadline | Lower = better |
| Convergence Speed | Episodes to stable reward | Lower = better |
| Resource Utilization | Avg CPU/memory usage | Higher = better |
| Fairness Index | Jain's Fairness Index | Higher = better |

---

## 📊 Baseline Algorithms

| Algorithm | Type | Reference |
|-----------|------|-----------|
| Round Robin | Heuristic | Classic |
| Min-Min | Heuristic | Arunarani et al. 2019 |
| DQN | Single-agent RL | — |
| PPO | Single-agent RL | — |
| IPPO | Independent MARL | — |
| **MARL-CTDE (Ours)** | Cooperative MARL | This paper |

---

## 🧪 Statistical Analysis

All results include:
- Wilcoxon signed-rank test (p < 0.05)
- 95% Confidence Intervals
- Cohen's d effect size
- 5 independent runs per experiment

---

## 📝 Citation

```bibtex
@article{[AUTHOR]2025marl,
  title={Cooperative Multi-Agent Reinforcement Learning with Centralized Training 
         and Decentralized Execution for Distributed Cloud Job Scheduling},
  author={[YOUR NAME et al.]},
  journal={Future Generation Computer Systems},
  year={2025},
  publisher={Elsevier},
  note={Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Google Borg Team for the cluster trace dataset
- CloudSim Plus for the simulation framework
