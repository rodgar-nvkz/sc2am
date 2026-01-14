import signal
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger

from scam.envs.impala_v2 import NUM_COMMANDS, OBS_SIZE, SC2GymEnv
from scam.impala_v2.config import IMPALAConfig
from scam.impala_v2.interop import SharedWeights
from scam.impala_v2.model import ActorCritic


@dataclass
class Rollout:
    """A single rollout from a worker with hybrid actions."""
    worker_id: int
    observations: np.ndarray           # (T, obs_size)
    commands: np.ndarray               # (T,) discrete command indices
    angles: np.ndarray                 # (T, 2) sin/cos angle encoding
    rewards: np.ndarray                # (T,)
    dones: np.ndarray                  # (T,) bool
    behavior_cmd_log_probs: np.ndarray # (T,) log probs of commands
    behavior_angle_log_probs: np.ndarray # (T,) log probs of angles
    behavior_values: np.ndarray        # (T,) value estimates from behavior policy
    next_observation: np.ndarray       # (obs_size,) for bootstrapping
    next_done: bool                    # Whether final state is terminal
    weight_version: int                # Version of weights used for this rollout
    episode_returns: list              # List of completed episode returns
    episode_lengths: list              # List of completed episode lengths


@dataclass
class RolloutBatch:
    """Pre-allocated batch of rollouts, ready to convert to tensors without stacking."""
    observations: np.ndarray           # (B, T, obs_size)
    commands: np.ndarray               # (B, T)
    angles: np.ndarray                 # (B, T, 2)
    rewards: np.ndarray                # (B, T)
    dones: np.ndarray                  # (B, T)
    behavior_cmd_log_probs: np.ndarray # (B, T)
    behavior_angle_log_probs: np.ndarray # (B, T)
    behavior_values: np.ndarray        # (B, T)
    next_observations: np.ndarray      # (B, obs_size)
    weight_versions: np.ndarray        # (B,) weight version used for each rollout

    episode_returns: list = field(default_factory=list)
    episode_lengths: list = field(default_factory=list)

    _count: int = 0
    _capacity: int = 0

    @classmethod
    def create(cls, batch_size: int, rollout_length: int, obs_size: int) -> "RolloutBatch":
        """Pre-allocate arrays for the full batch."""
        batch = cls(
            observations=np.empty((batch_size, rollout_length, obs_size), dtype=np.float32),
            commands=np.empty((batch_size, rollout_length), dtype=np.int64),
            angles=np.empty((batch_size, rollout_length, 2), dtype=np.float32),
            rewards=np.empty((batch_size, rollout_length), dtype=np.float32),
            dones=np.empty((batch_size, rollout_length), dtype=bool),
            behavior_cmd_log_probs=np.empty((batch_size, rollout_length), dtype=np.float32),
            behavior_angle_log_probs=np.empty((batch_size, rollout_length), dtype=np.float32),
            behavior_values=np.empty((batch_size, rollout_length), dtype=np.float32),
            next_observations=np.empty((batch_size, obs_size), dtype=np.float32),
            weight_versions=np.empty(batch_size, dtype=np.int64),
        )
        batch._capacity = batch_size
        return batch

    def insert(self, rollout: Rollout):
        """Insert a rollout at the next available slot."""
        i = self._count
        self.observations[i] = rollout.observations
        self.commands[i] = rollout.commands
        self.angles[i] = rollout.angles
        self.rewards[i] = rollout.rewards
        self.dones[i] = rollout.dones
        self.behavior_cmd_log_probs[i] = rollout.behavior_cmd_log_probs
        self.behavior_angle_log_probs[i] = rollout.behavior_angle_log_probs
        self.behavior_values[i] = rollout.behavior_values
        self.next_observations[i] = rollout.next_observation
        self.weight_versions[i] = rollout.weight_version

        self.episode_returns.extend(rollout.episode_returns)
        self.episode_lengths.extend(rollout.episode_lengths)
        self._count += 1

    def is_full(self) -> bool:
        return self._count >= self._capacity

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Convert to tensors - no stacking needed!"""
        return {
            "observations": torch.from_numpy(self.observations).to(device),
            "commands": torch.from_numpy(self.commands).to(device),
            "angles": torch.from_numpy(self.angles).to(device),
            "rewards": torch.from_numpy(self.rewards).to(device),
            "dones": torch.from_numpy(self.dones).to(device),
            "behavior_cmd_log_probs": torch.from_numpy(self.behavior_cmd_log_probs).to(device),
            "behavior_angle_log_probs": torch.from_numpy(self.behavior_angle_log_probs).to(device),
            "behavior_values": torch.from_numpy(self.behavior_values).to(device),
            "next_observations": torch.from_numpy(self.next_observations).to(device),
        }


@dataclass
class RolloutResult:
    """Result of collect_rollout including continuation state."""
    rollout: Rollout
    next_obs: np.ndarray          # Observation to continue from
    current_episode_return: float # Accumulated return in ongoing episode
    current_episode_length: int   # Steps in ongoing episode


def collector_worker(worker_id: int, rollout_queue: Any, shared_weights: SharedWeights, shutdown_event: Any, config: IMPALAConfig):
    """Worker process that runs autonomous rollouts."""

    # Ignore SIGINT in workers (let main handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        env = SC2GymEnv({"upgrade_level": config.upgrade_levels})
        model = ActorCritic(OBS_SIZE, NUM_COMMANDS)
        model.eval()

        obs, _ = env.reset()
        local_version = shared_weights.pull(model)
        ongoing_episode_length = 0
        ongoing_episode_return = 0.0

        while not shutdown_event.is_set():
            # Check for weight updates (async, between rollouts)
            current_version = shared_weights.get_version()
            if current_version > local_version:
                local_version = shared_weights.pull(model)
                logger.debug(f"Worker {worker_id} updated to weight version {local_version}")

            # Collect rollout
            result = collect_rollout(
                env=env,
                model=model,
                obs=obs,
                rollout_length=config.rollout_length,
                worker_id=worker_id,
                weight_version=local_version,
                ongoing_episode_return=ongoing_episode_return,
                ongoing_episode_length=ongoing_episode_length,
            )

            # Update state for next rollout
            obs = result.next_obs
            ongoing_episode_return = result.current_episode_return
            ongoing_episode_length = result.current_episode_length

            # Send rollout to main (non-blocking put with timeout)
            try:
                rollout_queue.put(result.rollout, timeout=1.0)
            except Exception as e:
                logger.warning(f"Worker {worker_id} failed to send rollout: {e}")
                continue

        logger.info(f"Worker {worker_id} shutting down...")
        env.close()

    except Exception as e:
        logger.error(f"Worker {worker_id} crashed: {e}")
        import traceback
        traceback.print_exc()


def collect_rollout(
    env,
    model: ActorCritic,
    obs: np.ndarray,
    rollout_length: int,
    worker_id: int,
    weight_version: int,
    ongoing_episode_return: float = 0.0,
    ongoing_episode_length: int = 0,
) -> RolloutResult:
    """Collect a single rollout of fixed length with hybrid actions."""

    obs_size = obs.shape[0]

    # Pre-allocate arrays (more efficient than list append -> convert)
    observations = np.empty((rollout_length, obs_size), dtype=np.float32)
    commands = np.empty(rollout_length, dtype=np.int64)
    angles = np.empty((rollout_length, 2), dtype=np.float32)
    rewards = np.empty(rollout_length, dtype=np.float32)
    dones = np.empty(rollout_length, dtype=bool)
    cmd_log_probs = np.empty(rollout_length, dtype=np.float32)
    angle_log_probs = np.empty(rollout_length, dtype=np.float32)
    values = np.empty(rollout_length, dtype=np.float32)

    episode_returns = []
    episode_lengths = []

    current_episode_return = ongoing_episode_return
    current_episode_length = ongoing_episode_length
    next_done = False

    with torch.no_grad():
        for t in range(rollout_length):
            observations[t] = obs

            # Get action from policy (autoregressive sampling)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            command, angle, cmd_log_prob, angle_log_prob, _, value = model.get_action_and_value(obs_tensor)

            command = command.squeeze(0)
            angle = angle.squeeze(0)
            cmd_log_prob = cmd_log_prob.squeeze(0)
            angle_log_prob = angle_log_prob.squeeze(0)
            value = value.squeeze(0) if value.dim() > 0 else value

            commands[t] = command.item()
            angles[t] = angle.numpy()
            cmd_log_probs[t] = cmd_log_prob.item()
            angle_log_probs[t] = angle_log_prob.item()
            values[t] = value.item()

            # Create hybrid action dict for environment
            action = {'command': command.item(), 'angle': angle.numpy()}

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards[t] = reward
            dones[t] = done

            current_episode_return += reward
            current_episode_length += 1

            # Handle episode end
            if done:
                episode_returns.append(current_episode_return)
                episode_lengths.append(current_episode_length)
                current_episode_return = 0.0
                current_episode_length = 0
                obs, _ = env.reset()
                next_done = True
            else:
                next_done = False

    # Get final observation for bootstrapping
    next_obs = obs

    rollout = Rollout(
        observations=observations,
        commands=commands,
        angles=angles,
        rewards=rewards,
        dones=dones,
        behavior_cmd_log_probs=cmd_log_probs,
        behavior_angle_log_probs=angle_log_probs,
        behavior_values=values,
        next_observation=next_obs.astype(np.float32),
        next_done=next_done,
        worker_id=worker_id,
        weight_version=weight_version,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )

    return RolloutResult(
        rollout=rollout,
        next_obs=next_obs,
        current_episode_return=current_episode_return,
        current_episode_length=current_episode_length,
    )
