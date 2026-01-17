"""
Simplified IMPALA collector with LSTM support.

Each worker collects complete episodes and sends them to the learner.
The LSTM hidden state is tracked during collection and reset at episode boundaries.
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
    actions: np.ndarray  # (T,) - discrete actions
    rewards: np.ndarray  # (T,)
    behavior_log_probs: np.ndarray  # (T,)
    behavior_values: np.ndarray  # (T,)
    weight_version: int
    won: bool  # True if marine survived and all zerglings dead

    @property
    def length(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())


@dataclass
class EpisodeBatch:
    """Batch of complete episodes for training.

    Episodes are padded to max length and batched together for efficient
    GPU processing. A mask indicates valid (non-padded) positions.
    """

    # Padded and batched tensors (num_episodes, max_len, ...)
    observations: np.ndarray  # (B, T_max, obs_size)
    actions: np.ndarray  # (B, T_max)
    rewards: np.ndarray  # (B, T_max)
    behavior_log_probs: np.ndarray  # (B, T_max)
    behavior_values: np.ndarray  # (B, T_max)
    vtrace_targets: np.ndarray  # (B, T_max)
    advantages: np.ndarray  # (B, T_max)

    # Mask for valid (non-padded) positions: True = valid
    mask: np.ndarray  # (B, T_max)

    # Metadata
    episode_lengths: list[int]
    episode_returns: list[float]
    episode_wins: list[bool]  # Track wins separately from rewards
    weight_versions: list[int]
    num_episodes: int
    total_steps: int
    max_length: int

    @classmethod
    def from_episodes(cls, episodes: list[Episode], config: IMPALAConfig) -> "EpisodeBatch":
        """Create batch from list of episodes, computing V-trace per episode.

        Episodes are padded to max length and stacked into batched tensors
        for efficient GPU processing.
        """
        num_episodes = len(episodes)
        episode_lengths = [ep.length for ep in episodes]
        episode_returns = [ep.total_reward for ep in episodes]
        episode_wins = [ep.won for ep in episodes]
        weight_versions = [ep.weight_version for ep in episodes]
        total_steps = sum(episode_lengths)
        max_length = max(episode_lengths)

        # Pre-allocate padded arrays
        obs_size = config.model.obs_size
        observations = np.zeros((num_episodes, max_length, obs_size), dtype=np.float32)
        actions = np.zeros((num_episodes, max_length), dtype=np.int64)
        rewards = np.zeros((num_episodes, max_length), dtype=np.float32)
        behavior_log_probs = np.zeros((num_episodes, max_length), dtype=np.float32)
        behavior_values = np.zeros((num_episodes, max_length), dtype=np.float32)
        vtrace_targets = np.zeros((num_episodes, max_length), dtype=np.float32)
        advantages = np.zeros((num_episodes, max_length), dtype=np.float32)
        mask = np.zeros((num_episodes, max_length), dtype=bool)

        # Fill arrays and compute V-trace per episode
        for i, ep in enumerate(episodes):
            T = ep.length

            # Copy episode data (rest stays zero-padded)
            observations[i, :T] = ep.observations
            actions[i, :T] = ep.actions
            rewards[i, :T] = ep.rewards
            behavior_log_probs[i, :T] = ep.behavior_log_probs
            behavior_values[i, :T] = ep.behavior_values
            mask[i, :T] = True

            # Compute V-trace for this episode
            ep_vtrace, ep_adv = compute_vtrace_episode(
                rewards=ep.rewards,
                values=ep.behavior_values,
                gamma=config.gamma
            )
            vtrace_targets[i, :T] = ep_vtrace
            advantages[i, :T] = ep_adv

        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            behavior_log_probs=behavior_log_probs,
            behavior_values=behavior_values,
            vtrace_targets=vtrace_targets,
            advantages=advantages,
            mask=mask,
            episode_lengths=episode_lengths,
            episode_returns=episode_returns,
            episode_wins=episode_wins,
            weight_versions=weight_versions,
            num_episodes=num_episodes,
            total_steps=total_steps,
            max_length=max_length,
        )

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Convert to GPU tensors for batched training.

        Returns batched tensors: (num_episodes, max_length, ...)
        """
        return {
            "observations": torch.from_numpy(self.observations).to(device),
            "actions": torch.from_numpy(self.actions).to(device),
            "rewards": torch.from_numpy(self.rewards).to(device),
            "behavior_log_probs": torch.from_numpy(self.behavior_log_probs).to(device),
            "behavior_values": torch.from_numpy(self.behavior_values).to(device),
            "vtrace_targets": torch.from_numpy(self.vtrace_targets).to(device),
            "advantages": torch.from_numpy(self.advantages).to(device),
            "mask": torch.from_numpy(self.mask).to(device),
        }


def compute_vtrace_episode(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute V-trace targets and advantages for a single complete episode.

    Since episode is complete (terminal state), bootstrap value = 0.
    Simplified version without importance sampling (on-policy assumption).
    """
    T = len(rewards)

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
    """Worker process that collects complete episodes with LSTM."""

    # Ignore SIGINT in workers (let main handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        env = SC2GymEnv({
            "upgrade_level": config.upgrade_levels,
            "num_move_directions": config.model.num_move_directions,
        })
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


def collect_episode(
    env,
    model: ActorCritic,
    worker_id: int,
    weight_version: int,
    max_steps: int = 1024
) -> Episode:
    """Collect a single complete episode with LSTM hidden state tracking."""

    # Use lists for variable-length episode
    observations = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    obs, info = env.reset()
    done = False
    steps = 0
    won = False  # Track if episode was won

    # Initialize LSTM hidden state
    hidden = model.get_initial_hidden(batch_size=1)

    with torch.no_grad():
        while not done and steps < max_steps:
            observations.append(obs)

            # Get action from policy (with LSTM)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            output = model(obs_tensor, hidden=hidden)

            # Update hidden state for next step
            hidden = output.hidden

            # Record action and statistics
            action = output.action.action.item()
            actions.append(action)
            log_probs.append(output.action.log_prob.item())
            values.append(output.value.item())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            won = info.get("won", False)
            steps += 1

    # Convert lists to arrays
    return Episode(
        worker_id=worker_id,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        behavior_log_probs=np.array(log_probs, dtype=np.float32),
        behavior_values=np.array(values, dtype=np.float32),
        weight_version=weight_version,
        won=won,
    )
