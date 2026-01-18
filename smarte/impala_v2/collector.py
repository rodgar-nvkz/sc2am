"""
Simplified IMPALA collector: 1 game = 1 episode.

Each worker collects complete episodes and sends them to the learner.
No padding, no fixed rollout length, no mid-episode resets.
"""

import signal
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger

from .config import IMPALAConfig
from .env import SC2GymEnv
from .interop import SharedWeights
from .model import ActorCritic


@dataclass
class Episode:
    """A single complete episode from a worker."""

    worker_id: int
    observations: np.ndarray  # (T, obs_size)
    commands: np.ndarray  # (T,)
    angles: np.ndarray  # (T, 2)
    rewards: np.ndarray  # (T,)
    behavior_cmd_log_probs: np.ndarray  # (T,)
    behavior_angle_log_probs: np.ndarray  # (T,)
    behavior_values: np.ndarray  # (T,)
    action_masks: np.ndarray  # (T, num_commands)
    weight_version: int

    @property
    def length(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())


@dataclass
class EpisodeBatch:
    """Batch of complete episodes, concatenated for training."""

    observations: np.ndarray  # (N_total, obs_size)
    commands: np.ndarray  # (N_total,)
    angles: np.ndarray  # (N_total, 2)
    rewards: np.ndarray  # (N_total,)
    behavior_cmd_log_probs: np.ndarray  # (N_total,)
    behavior_angle_log_probs: np.ndarray  # (N_total,)
    behavior_values: np.ndarray  # (N_total,)
    action_masks: np.ndarray  # (N_total, num_commands)

    # Per-episode V-trace results (computed separately, then concatenated)
    vtrace_targets: np.ndarray  # (N_total,)
    advantages: np.ndarray  # (N_total,)

    # Metadata
    episode_lengths: list[int]
    episode_returns: list[float]
    weight_versions: list[int]
    num_episodes: int

    @classmethod
    def from_episodes(cls, episodes: list[Episode], config: IMPALAConfig) -> "EpisodeBatch":
        """Create batch from list of episodes, computing V-trace per episode."""

        # Collect metadata
        episode_lengths = [ep.length for ep in episodes]
        episode_returns = [ep.total_reward for ep in episodes]
        weight_versions = [ep.weight_version for ep in episodes]
        total_steps = sum(episode_lengths)

        # Pre-allocate concatenated arrays
        observations = np.empty((total_steps, config.model.obs_size), dtype=np.float32)
        commands = np.empty(total_steps, dtype=np.int64)
        angles = np.empty((total_steps, 2), dtype=np.float32)
        rewards = np.empty(total_steps, dtype=np.float32)
        behavior_cmd_log_probs = np.empty(total_steps, dtype=np.float32)
        behavior_angle_log_probs = np.empty(total_steps, dtype=np.float32)
        behavior_values = np.empty(total_steps, dtype=np.float32)
        action_masks = np.empty((total_steps, config.model.num_commands), dtype=bool)
        vtrace_targets = np.empty(total_steps, dtype=np.float32)
        advantages = np.empty(total_steps, dtype=np.float32)

        # Fill arrays and compute V-trace per episode
        offset = 0
        for ep in episodes:
            T = ep.length
            end = offset + T

            # Copy episode data
            observations[offset:end] = ep.observations
            commands[offset:end] = ep.commands
            angles[offset:end] = ep.angles
            rewards[offset:end] = ep.rewards
            behavior_cmd_log_probs[offset:end] = ep.behavior_cmd_log_probs
            behavior_angle_log_probs[offset:end] = ep.behavior_angle_log_probs
            behavior_values[offset:end] = ep.behavior_values
            action_masks[offset:end] = ep.action_masks

            # Compute V-trace for this episode (terminal state, bootstrap = 0)
            ep_vtrace, ep_adv = compute_vtrace_episode(rewards=ep.rewards, values=ep.behavior_values, gamma=config.gamma)
            vtrace_targets[offset:end] = ep_vtrace
            advantages[offset:end] = ep_adv

            offset = end

        return cls(
            observations=observations,
            commands=commands,
            angles=angles,
            rewards=rewards,
            behavior_cmd_log_probs=behavior_cmd_log_probs,
            behavior_angle_log_probs=behavior_angle_log_probs,
            behavior_values=behavior_values,
            action_masks=action_masks,
            vtrace_targets=vtrace_targets,
            advantages=advantages,
            episode_lengths=episode_lengths,
            episode_returns=episode_returns,
            weight_versions=weight_versions,
            num_episodes=len(episodes),
        )

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Convert to tensors for training."""
        return {
            "observations": torch.from_numpy(self.observations).to(device),
            "commands": torch.from_numpy(self.commands).to(device),
            "angles": torch.from_numpy(self.angles).to(device),
            "rewards": torch.from_numpy(self.rewards).to(device),
            "behavior_cmd_log_probs": torch.from_numpy(self.behavior_cmd_log_probs).to(device),
            "behavior_angle_log_probs": torch.from_numpy(self.behavior_angle_log_probs).to(device),
            "behavior_values": torch.from_numpy(self.behavior_values).to(device),
            "action_masks": torch.from_numpy(self.action_masks).to(device),
            "vtrace_targets": torch.from_numpy(self.vtrace_targets).to(device),
            "advantages": torch.from_numpy(self.advantages).to(device),
        }


def compute_vtrace_episode(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute V-trace targets and advantages for a single complete episode.

    Since episode is complete (terminal state), bootstrap value = 0.
    Simplified version without importance sampling (on-policy assumption). Avg staleness < 3.
    """
    T = len(rewards)

    # TD targets: standard TD(0) since episode is complete
    # V_target[t] = r[t] + gamma * V_target[t+1], with V_target[T] = 0
    vtrace_targets = np.zeros(T, dtype=np.float32)
    advantages = np.zeros(T, dtype=np.float32)

    # Backward pass
    next_value = 0.0  # Terminal state
    for t in reversed(range(T)):
        vtrace_targets[t] = rewards[t] + gamma * next_value
        advantages[t] = vtrace_targets[t] - values[t]
        next_value = vtrace_targets[t]

    return vtrace_targets, advantages


def collector_worker(
    worker_id: int,
    episode_queue: Any,
    shared_weights: SharedWeights,
    shutdown_event: Any,
    config: IMPALAConfig,
):
    """Worker process that collects complete episodes."""

    # Ignore SIGINT in workers (let main handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        env = SC2GymEnv({"upgrade_level": config.upgrade_levels})
        model = ActorCritic(config.model)
        model.eval()

        local_version = shared_weights.pull(model)

        while not shutdown_event.is_set():
            # Check for weight updates before each episode
            current_version = shared_weights.get_version()
            if current_version > local_version:
                local_version = shared_weights.pull(model)
                logger.debug(f"Worker {worker_id} updated to weight version {local_version}")

            # Collect one complete episode
            episode = collect_episode(
                env=env,
                model=model,
                worker_id=worker_id,
                weight_version=local_version,
                max_steps=config.max_episode_steps,
            )

            # Send episode to main
            try:
                episode_queue.put(episode, timeout=1.0)
            except Exception as e:
                logger.warning(f"Worker {worker_id} failed to send episode: {e}")
                continue

        logger.info(f"Worker {worker_id} shutting down...")
        env.close()

    except Exception as e:
        logger.error(f"Worker {worker_id} crashed: {e}")
        import traceback

        traceback.print_exc()


def collect_episode(env, model: ActorCritic, worker_id: int, weight_version: int, max_steps: int = 1024) -> Episode:
    """Collect a single complete episode."""

    # Use lists for variable-length episode
    observations = []
    commands = []
    angles = []
    rewards = []
    cmd_log_probs = []
    angle_log_probs = []
    values = []
    action_masks = []

    obs, info = env.reset()
    action_mask = info["action_mask"]
    done = False
    steps = 0

    with torch.no_grad():
        while not done and steps < max_steps:
            observations.append(obs)
            action_masks.append(action_mask)

            # Get action from policy
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            mask_tensor = torch.from_numpy(action_mask).unsqueeze(0)
            output = model(obs_tensor, action_mask=mask_tensor)

            commands.append(output.command.action.item())
            angles.append(output.angle.action.squeeze(0).numpy())
            cmd_log_probs.append(output.command.log_prob.item())
            angle_log_probs.append(output.angle.log_prob.item())
            values.append(output.value.item())

            # Step environment
            action = {"command": output.command.action.item(), "angle": output.angle.action.squeeze(0).numpy()}
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            action_mask = info["action_mask"]
            steps += 1

    # Convert lists to arrays
    return Episode(
        worker_id=worker_id,
        observations=np.array(observations, dtype=np.float32),
        commands=np.array(commands, dtype=np.int64),
        angles=np.array(angles, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        behavior_cmd_log_probs=np.array(cmd_log_probs, dtype=np.float32),
        behavior_angle_log_probs=np.array(angle_log_probs, dtype=np.float32),
        behavior_values=np.array(values, dtype=np.float32),
        action_masks=np.array(action_masks, dtype=bool),
        weight_version=weight_version,
    )
