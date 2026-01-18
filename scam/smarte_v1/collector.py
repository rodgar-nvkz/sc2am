"""
IMPALA collector with hybrid action space support.

Each worker collects complete episodes and sends them to the learner.
The GRU hidden state is tracked during collection and reset at episode boundaries.

Action Space:
    - action_type: Discrete [MOVE=0, ATTACK=1, STOP=2]
    - move_direction: Continuous [sin, cos] (used when action_type=MOVE)
    - attack_target: Integer index (used when action_type=ATTACK)

Environment expects:
    - command: Discrete(3) - MOVE=0, ATTACK_Z1=1, ATTACK_Z2=2
    - angle: Box(2) - [sin, cos] for movement direction
"""

import os
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
    """A single complete episode from a worker.

    Stores all action components for hybrid action space training.
    """

    worker_id: int
    observations: np.ndarray  # (T, obs_size)
    action_types: np.ndarray  # (T,) - discrete action type indices
    move_directions: np.ndarray  # (T, 2) - [sin, cos] for each step
    attack_targets: np.ndarray  # (T,) - target indices for each step
    action_masks: np.ndarray  # (T, 3) - valid action mask per step
    rewards: np.ndarray  # (T,)
    behavior_log_probs: np.ndarray  # (T,) - combined log probs
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
    action_types: np.ndarray  # (B, T_max)
    move_directions: np.ndarray  # (B, T_max, 2)
    attack_targets: np.ndarray  # (B, T_max)
    action_masks: np.ndarray  # (B, T_max, 3)
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
    episode_wins: list[bool]
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
        num_action_types = config.model.num_action_types

        observations = np.zeros((num_episodes, max_length, obs_size), dtype=np.float32)
        action_types = np.zeros((num_episodes, max_length), dtype=np.int64)
        move_directions = np.zeros((num_episodes, max_length, 2), dtype=np.float32)
        attack_targets = np.zeros((num_episodes, max_length), dtype=np.int64)
        action_masks = np.ones((num_episodes, max_length, num_action_types), dtype=bool)
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
            action_types[i, :T] = ep.action_types
            move_directions[i, :T] = ep.move_directions
            attack_targets[i, :T] = ep.attack_targets
            action_masks[i, :T] = ep.action_masks
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
            action_types=action_types,
            move_directions=move_directions,
            attack_targets=attack_targets,
            action_masks=action_masks,
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
            "action_types": torch.from_numpy(self.action_types).to(device),
            "move_directions": torch.from_numpy(self.move_directions).to(device),
            "attack_targets": torch.from_numpy(self.attack_targets).to(device),
            "action_masks": torch.from_numpy(self.action_masks).to(device),
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
    """Worker process that collects complete episodes with GRU hidden state."""

    # Ignore SIGINT in workers (let main handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # === PERFORMANCE OPTIMIZATIONS ===
    # Limit PyTorch to 1 thread per worker to avoid CPU oversubscription
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Also set OpenMP/MKL threads via environment (may already be set by parent)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    try:
        env = SC2GymEnv()
        model = ActorCritic(config.model)
        model.eval()

        local_version = shared_weights.pull(model)
        episodes_since_sync = 0

        while not shutdown_event.is_set():
            # Check for weight updates periodically (not every episode)
            # This reduces lock contention significantly
            episodes_since_sync += 1
            if episodes_since_sync >= config.weight_sync_interval:
                current_version = shared_weights.get_version()
                if current_version > local_version:
                    local_version = shared_weights.pull(model)
                    logger.debug(f"Worker {worker_id} updated to weight version {local_version}")
                episodes_since_sync = 0

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
                episode_queue.put(episode, timeout=10.0)
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
    """Collect a single complete episode with GRU hidden state tracking.

    Handles hybrid action space:
        - Model outputs: action_type, move_direction, attack_target
        - Env expects: {"command": int, "angle": np.array}
    """

    # Use lists for variable-length episode
    observations = []
    action_types = []
    move_directions = []
    attack_targets = []
    action_masks = []
    rewards = []
    log_probs = []
    values = []

    obs, info = env.reset()
    done = False
    steps = 0
    won = False

    # Initialize GRU hidden state
    hidden = model.get_initial_hidden(batch_size=1)

    # Pre-allocate observation tensor for reuse
    obs_tensor = torch.empty(1, obs.shape[0], dtype=torch.float32)

    # Use inference_mode for faster execution
    with torch.inference_mode():
        while not done and steps < max_steps:
            observations.append(obs.copy())

            # Get action mask from environment
            # Env mask: [MOVE, ATTACK_Z1, ATTACK_Z2, STOP] (4 commands)
            # Model mask: [MOVE, ATTACK, STOP] (3 action types)
            env_action_mask = info.get("action_mask", None)
            action_mask_tensor = None
            if env_action_mask is not None:
                # Convert env mask [MOVE, ATTACK_Z1, ATTACK_Z2, STOP] to model mask [MOVE, ATTACK, STOP]
                # ATTACK is valid if either ATTACK_Z1 or ATTACK_Z2 is valid
                model_mask = np.array([
                    env_action_mask[0],  # MOVE
                    env_action_mask[1] or env_action_mask[2],  # ATTACK (either target valid)
                    env_action_mask[3] if len(env_action_mask) > 3 else True,  # STOP
                ], dtype=bool)
                action_masks.append(model_mask.copy())
                action_mask_tensor = torch.from_numpy(model_mask).unsqueeze(0)  # (1, 3)

                # Also create range mask for attack targeting (which enemies are in range)
                range_mask = torch.tensor([[env_action_mask[1], env_action_mask[2]]], dtype=torch.bool)
            else:
                action_masks.append(np.ones(3, dtype=bool))
                range_mask = None

            # Get action from policy
            obs_tensor[0].copy_(torch.from_numpy(obs))
            output = model(
                obs_tensor,
                hidden=hidden,
                action_mask=action_mask_tensor,
                range_mask=range_mask,
            )

            # Update hidden state for next step
            hidden = output.hidden

            # Record action components
            action_type = output.action.action_type[0].item()
            move_dir = output.action.move_direction[0].cpu().numpy()
            attack_target = output.action.attack_target[0].item()

            action_types.append(action_type)
            move_directions.append(move_dir.copy())
            attack_targets.append(attack_target)

            # Record statistics (combined log prob from model)
            log_probs.append(output.log_prob[0].item())
            values.append(output.value[0].item())

            # Convert to environment action format
            env_action = model.to_env_action(output.action, batch_idx=0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            rewards.append(reward)
            won = info.get("won", False)
            steps += 1

    # Convert lists to arrays
    return Episode(
        worker_id=worker_id,
        observations=np.array(observations, dtype=np.float32),
        action_types=np.array(action_types, dtype=np.int64),
        move_directions=np.array(move_directions, dtype=np.float32),
        attack_targets=np.array(attack_targets, dtype=np.int64),
        action_masks=np.array(action_masks, dtype=bool),
        rewards=np.array(rewards, dtype=np.float32),
        behavior_log_probs=np.array(log_probs, dtype=np.float32),
        behavior_values=np.array(values, dtype=np.float32),
        weight_version=weight_version,
        won=won,
    )
