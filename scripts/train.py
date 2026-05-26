"""
scripts/train.py
Main Training Script — MARL-CTDE Cloud Job Scheduling

Usage:
    python scripts/train.py --config configs/borg_config.yaml
    python scripts/train.py --config configs/borg_config.yaml --scale large
    python scripts/train.py --config configs/borg_config.yaml --seed 42

Outputs:
    results/logs/          — TensorBoard logs
    results/checkpoints/   — Model checkpoints
    results/metrics.json   — Training metrics
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agents.marl_agent import MARLCTDEAgent
from src.environment.cloud_env import CloudSchedulingEnv
from src.environment.borg_loader import BorgTraceLoader


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config: dict, args) -> dict:
    """Main training loop."""

    # ── Setup ──────────────────────────────────────────────────────────────
    seed = config['training']['seed']
    set_seed(seed)

    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)

    writer = SummaryWriter(log_dir=config['logging']['log_dir'])
    logger.info(f"Training MARL-CTDE | Seed: {seed} | Scale: {args.scale}")

    # ── Load Dataset ────────────────────────────────────────────────────────
    loader = BorgTraceLoader(
        data_path=config['dataset']['path'],
        seed=seed
    )

    logger.info("Loading Google Borg Trace 2019...")
    all_jobs = []
    for cell_id in config['dataset']['cell_ids'][:2]:  # Load first 2 cells
        cell_jobs = loader.load_cell(cell_id)
        all_jobs.extend(cell_jobs)

    logger.info(f"Total jobs loaded: {len(all_jobs)}")

    # Get scale config
    scale_config = config['scales'][args.scale]
    n_jobs_per_episode = scale_config['n_jobs']
    config['environment']['n_nodes'] = scale_config['n_nodes']

    # Create episode batches
    episode_batches = loader.create_episode_batches(
        all_jobs,
        n_jobs=n_jobs_per_episode,
        n_episodes=config['training']['n_episodes']
    )

    # ── Initialize Environment & Agent ──────────────────────────────────────
    env = CloudSchedulingEnv(config)
    agent = MARLCTDEAgent(config)

    # ── Training Loop ───────────────────────────────────────────────────────
    best_reward = float('-inf')
    episode_rewards = []
    training_metrics = {
        'episode_rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'sla_violation_rates': [],
        'resource_utilizations': []
    }

    pbar = tqdm(range(config['training']['n_episodes']), desc="Training")

    for episode in pbar:
        # Reset environment with new job batch
        jobs = episode_batches[episode % len(episode_batches)]
        obs_list, info = env.reset(options={'jobs': jobs})
        global_state = info['global_state']

        episode_reward = 0
        episode_sla = 0
        episode_util = 0
        steps = 0

        done = False
        while not done and steps < config['training']['max_steps_per_episode']:

            # Select actions (decentralized)
            action_masks = [env.get_action_mask(i) for i in range(env.n_agents)]
            actions, log_probs = agent.select_actions(obs_list, action_masks)

            # Step environment
            next_obs_list, rewards, terminated, truncated, step_info = env.step(actions.tolist())
            next_global_state = step_info['global_state']
            done = terminated or truncated

            # Store transition in replay buffer
            agent.buffer.add(
                obs=np.stack(obs_list),
                global_state=global_state,
                actions=actions,
                log_probs=log_probs,
                rewards=np.array(rewards),
                next_obs=np.stack(next_obs_list),
                next_global_state=next_global_state,
                dones=np.array([float(done)] * env.n_agents)
            )

            obs_list = next_obs_list
            global_state = next_global_state
            episode_reward += np.mean(rewards)
            episode_sla += step_info['sla_violations']
            episode_util += step_info['metrics'].resource_utilization
            steps += 1

        # ── Update agent ───────────────────────────────────────────────────
        update_metrics = {}
        if episode % config['training']['update_frequency'] == 0:
            update_metrics = agent.update()

        # ── Logging ────────────────────────────────────────────────────────
        episode_rewards.append(episode_reward)
        n_jobs = max(1, len(jobs))
        sla_rate = episode_sla / n_jobs
        util_rate = episode_util / max(1, steps)

        training_metrics['episode_rewards'].append(episode_reward)
        training_metrics['sla_violation_rates'].append(sla_rate)
        training_metrics['resource_utilizations'].append(util_rate)

        if update_metrics:
            training_metrics['actor_losses'].append(update_metrics.get('actor_loss', 0))
            training_metrics['critic_losses'].append(update_metrics.get('critic_loss', 0))

        # TensorBoard
        writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
        writer.add_scalar('Train/SLAViolationRate', sla_rate, episode)
        writer.add_scalar('Train/ResourceUtilization', util_rate, episode)
        if update_metrics:
            writer.add_scalar('Train/ActorLoss', update_metrics.get('actor_loss', 0), episode)
            writer.add_scalar('Train/CriticLoss', update_metrics.get('critic_loss', 0), episode)

        # Progress bar
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
        pbar.set_postfix({
            'reward': f"{avg_reward:.3f}",
            'sla': f"{sla_rate:.3f}",
            'util': f"{util_rate:.3f}"
        })

        # ── Checkpointing ──────────────────────────────────────────────────
        if episode % config['logging']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f"checkpoint_ep{episode}.pt"
            )
            agent.save(checkpoint_path)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(config['logging']['checkpoint_dir'], "best_model.pt"))

    # ── Save final metrics ─────────────────────────────────────────────────
    metrics_path = "results/training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)

    writer.close()
    logger.info(f"Training complete. Best reward: {best_reward:.4f}")
    logger.info(f"Results saved to results/")

    return training_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MARL-CTDE Cloud Scheduler")
    parser.add_argument('--config', type=str, default='configs/borg_config.yaml')
    parser.add_argument('--scale', type=str, default='small',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config['training']['seed'] = args.seed

    train(config, args)
