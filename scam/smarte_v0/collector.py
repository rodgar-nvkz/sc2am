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

    @property
    def length(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())


@dataclass
class EpisodeBatch:
    """Batch of complete episodes for training.

    For LSTM training, we keep episodes separate and process them
    sequentially through the network.
    """

    # List of episodes (not concatenated - needed for LSTM)
    episodes: list[Episode]

    # Pre-computed V-trace targets and advantages per episode
    vtrace_targets: list[np.ndarray]  # List of (T,) arrays
    advantages: list[np.ndarray]  # List of (T,) arrays

    # Metadata
    episode_lengths: list[int]
    episode_returns: list[float]
    weight_versions: list[int]
    num_episodes: int
    total_steps: int

    @classmethod
    def from_episodes(cls, episodes: list[Episode], config: IMPALAConfig) -> "EpisodeBatch":
        """Create batch from list of episodes, computing V-trace per episode."""

        # Collect metadata
        episode_lengths = [ep.length for ep in episodes]
        episode_returns = [ep.total_reward for ep in episodes]
        weight_versions = [ep.weight_version for ep in episodes]
        total_steps = sum(episode_lengths)

        # Compute V-trace for each episode
        vtrace_targets = []
        advantages = []

        for ep in episodes:
            ep_vtrace, ep_adv = compute_vtrace_episode(
                rewards=ep.rewards,
                values=ep.behavior_values,
                gamma=config.gamma
            )
            vtrace_targets.append(ep_vtrace)
            advantages.append(ep_adv)

        return cls(
            episodes=episodes,
            vtrace_targets=vtrace_targets,
            advantages=advantages,
            episode_lengths=episode_lengths,
            episode_returns=episode_returns,
            weight_versions=weight_versions,
            num_episodes=len(episodes),
            total_steps=total_steps,
        )

    def to_tensors(self, device: torch.device) -> dict[str, list[torch.Tensor]]:
        """Convert to tensors for training.

        Returns lists of tensors (one per episode) for LSTM processing.
        """
        return {
            "observations": [
                torch.from_numpy(ep.observations).to(device)
                for ep in self.episodes
            ],
            "actions": [
                torch.from_numpy(ep.actions).to(device)
                for ep in self.episodes
            ],
            "rewards": [
                torch.from_numpy(ep.rewards).to(device)
                for ep in self.episodes
            ],
            "behavior_log_probs": [
                torch.from_numpy(ep.behavior_log_probs).to(device)
                for ep in self.episodes
            ],
            "behavior_values": [
                torch.from_numpy(ep.behavior_values).to(device)
                for ep in self.episodes
            ],
            "vtrace_targets": [
                torch.from_numpy(vt).to(device)
                for vt in self.vtrace_targets
            ],
            "advantages": [
                torch.from_numpy(adv).to(device)
                for adv in self.advantages
            ],
        }

    def to_flat_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Convert to flat tensors (concatenated across episodes).

        This is used after forward_sequence to compute losses on all steps.
        """
        return {
            "observations": torch.from_numpy(
                np.concatenate([ep.observations for ep in self.episodes], axis=0)
            ).to(device),
            "actions": torch.from_numpy(
                np.concatenate([ep.actions for ep in self.episodes], axis=0)
            ).to(device),
            "rewards": torch.from_numpy(
                np.concatenate([ep.rewards for ep in self.episodes], axis=0)
            ).to(device),
            "behavior_log_probs": torch.from_numpy(
                np.concatenate([ep.behavior_log_probs for ep in self.episodes], axis=0)
            ).to(device),
            "behavior_values": torch.from_numpy(
                np.concatenate([ep.behavior_values for ep in self.episodes], axis=0)
            ).to(device),
            "vtrace_targets": torch.from_numpy(
                np.concatenate(self.vtrace_targets, axis=0)
            ).to(device),
            "advantages": torch.from_numpy(
                np.concatenate(self.advantages, axis=0)
            ).to(device),
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
    )
