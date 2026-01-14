"""
IMPALA-style training for SC2 Marine vs Zerglings environment.

This implements fully autonomous workers with:
- Local policy copies for inference (no per-step IPC)
- Async weight synchronization from main process
- V-trace off-policy correction
- Rollout-based communication (1024 steps per rollout)

Hybrid Action Space (Autoregressive):
- Discrete commands: STAY, MOVE, ATTACK_Z1, ATTACK_Z2
- Continuous angle (sin, cos) for MOVE command
- Angle head is conditioned on the discrete command via embedding

Architecture:
    Worker 1: [SC2 + Policy] -> rollouts -> Queue ->
    Worker 2: [SC2 + Policy] -> rollouts -> Queue ->  Main: V-trace -> Train -> Broadcast weights
    Worker N: [SC2 + Policy] -> rollouts -> Queue ->
                                              ^
                                              |
                                    Shared memory weights

Usage:
    python -m scam.train.impala train --steps 1000000 --num-workers 8
    python -m scam.train.impala eval --games 10 --model artifacts/models/impala_v2_xxx.pt
"""

import argparse
import ctypes
import glob
import multiprocessing as mp
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Event, Process, Queue, Value
from multiprocessing.sharedctypes import RawArray
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from scam.envs.impala_v2 import ACTION_MOVE, NUM_COMMANDS, OBS_SIZE, SC2GymEnv
from scam.settings import PROJECT_ROOT

# ============================================================================
# Hyperparameters
# ============================================================================

@dataclass
class IMPALAConfig:
    """IMPALA training configuration."""
    # Rollout settings
    rollout_length: int = 1024

    # Training settings
    num_workers: int = 8
    total_frames: int = 1_000_000
    mini_batch_size: int = 512
    num_epochs: int = 2

    # V-trace parameters
    gamma: float = 0.99
    rho_bar: float = 1.0  # Truncation for importance weights
    c_bar: float = 1.0    # Truncation for trace coefficients

    # PPO-style clipping
    clip_epsilon: float = 0.2

    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 40.0

    # Optimizer
    lr: float = 5e-4
    lr_eps: float = 1e-4
    lr_start_factor: float = 1.0
    lr_end_factor: float = 0.2

    # Environment
    upgrade_levels: list = field(default_factory=list)
    game_steps_per_env: list = field(default_factory=list)

    def __post_init__(self):
        self.upgrade_levels = self.upgrade_levels or [1, 2]
        self.game_steps_per_env = self.game_steps_per_env or [2, 4]


# ============================================================================
# Autoregressive Actor-Critic Network
# ============================================================================

class ActorCritic(nn.Module):
    """
    Autoregressive actor-critic for hybrid discrete-continuous action space.

    Action space:
    - Discrete command: STAY, MOVE, ATTACK_Z1, ATTACK_Z2 (4 options)
    - Continuous angle: (sin, cos) for movement direction (only used when command=MOVE)

    The angle head is conditioned on the discrete command via embedding.
    This allows the network to learn command-specific angle distributions.

    P(action | obs) = P(command | obs) * P(angle | obs, command)
    """

    def __init__(self, obs_size: int = OBS_SIZE, num_commands: int = NUM_COMMANDS):
        super().__init__()
        self.num_commands = num_commands

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        # Discrete command head
        self.command_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, num_commands),
        )

        # Command embedding (for conditioning angle head)
        self.command_embed = nn.Embedding(num_commands, 16)

        # Angle head (conditioned on command via embedding)
        # Outputs mean of (sin, cos) for movement direction
        self.angle_head = nn.Sequential(
            nn.Linear(64 + 16, 32),  # features + command embedding
            nn.Tanh(),
            nn.Linear(32, 2),  # sin, cos mean
        )

        # Learnable log std for angle (shared across all states)
        # Start with moderate variance for exploration
        self.angle_logstd = nn.Parameter(torch.tensor([-0.5, -0.5]))

        # Critic head (only depends on observation, not action)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights (orthogonal like PPO)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)

        # Smaller init for output layers
        cmd_out = self.command_head[-1]
        angle_out = self.angle_head[-1]
        critic_out = self.critic[-1]
        assert isinstance(cmd_out, nn.Linear)
        assert isinstance(angle_out, nn.Linear)
        assert isinstance(critic_out, nn.Linear)
        nn.init.orthogonal_(cmd_out.weight, gain=0.01)
        nn.init.orthogonal_(angle_out.weight, gain=0.01)
        nn.init.orthogonal_(critic_out.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """Forward pass returning features and command logits."""
        features = self.shared(obs)
        cmd_logits = self.command_head(features)
        value = self.critic(features)
        return features, cmd_logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.shared(obs)
        return self.critic(features).squeeze(-1)

    def get_angle_distribution(self, features: torch.Tensor, command: torch.Tensor):
        """Get angle distribution conditioned on command.

        Args:
            features: Shared features from encoder (B, 64)
            command: Discrete command indices (B,)

        Returns:
            Normal distribution over (sin, cos) angle encoding
        """
        cmd_embed = self.command_embed(command)  # (B, 16)
        angle_input = torch.cat([features, cmd_embed], dim=-1)  # (B, 64+16)
        angle_mean = self.angle_head(angle_input)  # (B, 2)

        # Normalize to unit circle for stability (optional but helps)
        angle_mean = F.normalize(angle_mean, dim=-1)

        angle_std = self.angle_logstd.exp()
        return torch.distributions.Normal(angle_mean, angle_std)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        command: Optional[torch.Tensor] = None,
        angle: Optional[torch.Tensor] = None,
    ):
        """
        Get action, log_prob, entropy, and value for given observation.

        If command/angle are provided, computes log_prob for those actions.
        Otherwise, samples new actions.

        Returns:
            command: (B,) discrete command indices
            angle: (B, 2) sin/cos angle encoding
            cmd_log_prob: (B,) log prob of command
            angle_log_prob: (B,) log prob of angle (sum over sin/cos dims)
            entropy: (B,) total entropy (command + angle)
            value: (B,) value estimates
        """
        features, cmd_logits, value = self.forward(obs)

        # Command distribution
        cmd_dist = torch.distributions.Categorical(logits=cmd_logits)

        if command is None:
            command = cmd_dist.sample()
        cmd_log_prob = cmd_dist.log_prob(command)
        cmd_entropy = cmd_dist.entropy()

        # Angle distribution (conditioned on command)
        angle_dist = self.get_angle_distribution(features, command)

        if angle is None:
            angle = angle_dist.sample()
            # Normalize sampled angle to unit circle
            angle = F.normalize(angle, dim=-1)

        # Log prob for angle (sum over sin/cos dimensions)
        angle_log_prob = angle_dist.log_prob(angle).sum(dim=-1)
        angle_entropy = angle_dist.entropy().sum(dim=-1)

        # Total entropy
        entropy = cmd_entropy + angle_entropy

        return command, angle, cmd_log_prob, angle_log_prob, entropy, value.squeeze(-1)

    def get_deterministic_action(self, obs: torch.Tensor):
        """Get deterministic action for evaluation (argmax command, mean angle)."""
        features, cmd_logits, _ = self.forward(obs)

        # Deterministic command (argmax)
        command = cmd_logits.argmax(dim=-1)

        # Deterministic angle (mean of distribution)
        angle_dist = self.get_angle_distribution(features, command)
        angle = F.normalize(angle_dist.mean, dim=-1)

        return command, angle


# ============================================================================
# Shared Weights (for async weight distribution)
# ============================================================================

class SharedWeights:
    """Shared memory for model weights with version tracking."""

    def __init__(self, model: nn.Module):
        # Flatten all parameters
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        self.size = params.numel()

        # Create shared memory array
        self.shared_array = RawArray(ctypes.c_float, self.size)

        # Version counter for staleness tracking
        self.version = Value(ctypes.c_int, 0)

        # Initialize with model weights
        self.push(model)

    def push(self, model: nn.Module):
        """Copy model weights to shared memory."""
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        np_array = np.frombuffer(self.shared_array, dtype=np.float32)
        np.copyto(np_array, params.detach().numpy())

        with self.version.get_lock():
            self.version.value += 1

    def pull(self, model: nn.Module) -> int:
        """Copy weights from shared memory to model. Returns version."""
        np_array = np.frombuffer(self.shared_array, dtype=np.float32).copy()
        params = torch.from_numpy(np_array)
        torch.nn.utils.vector_to_parameters(params, model.parameters())

        with self.version.get_lock():
            return self.version.value

    def get_version(self) -> int:
        with self.version.get_lock():
            return self.version.value


# ============================================================================
# Rollout Data Structure
# ============================================================================

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
class RolloutResult:
    """Result of collect_rollout including continuation state."""
    rollout: Rollout
    next_obs: np.ndarray          # Observation to continue from
    current_episode_return: float # Accumulated return in ongoing episode
    current_episode_length: int   # Steps in ongoing episode


# ============================================================================
# Worker Process
# ============================================================================

def worker_process(
    worker_id: int,
    rollout_queue: Any,
    shared_weights: "SharedWeights",
    shutdown_event: Any,
    config: "IMPALAConfig",
):
    """Worker process that runs autonomous rollouts."""

    # Ignore SIGINT in workers (let main handle it)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger.info(f"Worker {worker_id} starting...")

    try:
        # Create local environment (simple Gym-style API)
        env = SC2GymEnv({
            "upgrade_level": config.upgrade_levels,
            "game_steps_per_env": config.game_steps_per_env,
        })

        # Create local model
        model = ActorCritic()
        model.eval()

        # Pull initial weights
        local_version = shared_weights.pull(model)
        logger.info(f"Worker {worker_id} initialized with weight version {local_version}")

        # Initialize environment
        obs, _ = env.reset()

        # Track ongoing episode stats across rollouts
        ongoing_episode_return = 0.0
        ongoing_episode_length = 0

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

    observations = []
    commands = []
    angles = []
    rewards = []
    dones = []
    cmd_log_probs = []
    angle_log_probs = []
    values = []
    episode_returns = []
    episode_lengths = []

    current_episode_return = ongoing_episode_return
    current_episode_length = ongoing_episode_length
    next_done = False

    with torch.no_grad():
        for _ in range(rollout_length):
            observations.append(obs)

            # Get action from policy (autoregressive sampling)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            command, angle, cmd_log_prob, angle_log_prob, _, value = model.get_action_and_value(obs_tensor)

            command = command.squeeze(0)
            angle = angle.squeeze(0)
            cmd_log_prob = cmd_log_prob.squeeze(0)
            angle_log_prob = angle_log_prob.squeeze(0)
            value = value.squeeze(0) if value.dim() > 0 else value

            commands.append(command.item())
            angles.append(angle.numpy())
            cmd_log_probs.append(cmd_log_prob.item())
            angle_log_probs.append(angle_log_prob.item())
            values.append(value.item())

            # Create hybrid action dict for environment
            action = {'command': command.item(), 'angle': angle.numpy()}

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            dones.append(done)

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
        observations=np.array(observations, dtype=np.float32),
        commands=np.array(commands, dtype=np.int64),
        angles=np.array(angles, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        behavior_cmd_log_probs=np.array(cmd_log_probs, dtype=np.float32),
        behavior_angle_log_probs=np.array(angle_log_probs, dtype=np.float32),
        behavior_values=np.array(values, dtype=np.float32),
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


# ============================================================================
# V-trace Implementation
# ============================================================================

def compute_vtrace(
    behavior_log_probs: torch.Tensor,  # (B, T) combined log probs
    target_log_probs: torch.Tensor,    # (B, T) combined log probs
    rewards: torch.Tensor,             # (B, T)
    values: torch.Tensor,              # (B, T) current value estimates
    bootstrap_value: torch.Tensor,     # (B,) value at T+1
    dones: torch.Tensor,               # (B, T)
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute V-trace targets and advantages.

    Returns:
        vtrace_targets: (B, T) V-trace value targets
        advantages: (B, T) V-trace advantages for policy gradient
    """
    B, T = rewards.shape
    device = rewards.device

    # Compute importance sampling ratios
    log_rhos = target_log_probs - behavior_log_probs
    rhos = torch.exp(log_rhos)

    # Truncate ratios
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    # Compute TD errors: δ_t = ρ_t * (r_t + γ * V(s_{t+1}) - V(s_t))
    # Handle terminal states
    not_done = (~dones).float()

    # Shift values for V(s_{t+1})
    next_values = torch.zeros_like(values)
    next_values[:, :-1] = values[:, 1:]
    next_values[:, -1] = bootstrap_value

    # Mask next values at episode boundaries
    next_values = next_values * not_done

    # TD errors
    deltas = clipped_rhos * (rewards + gamma * next_values - values)

    # Compute V-trace targets using backward recursion
    # v_s - V(s) = δ_s + γ * c_s * (v_{s+1} - V(s_{s+1}))
    vtrace_minus_v = torch.zeros(B, T + 1, device=device)

    for t in reversed(range(T)):
        vtrace_minus_v[:, t] = deltas[:, t] + gamma * cs[:, t] * not_done[:, t] * vtrace_minus_v[:, t + 1]

    vtrace_targets = vtrace_minus_v[:, :-1] + values

    # Advantages for policy gradient: ρ_t * (r_t + γ * v_{t+1} - V(s_t))
    # Using V-trace targets for v_{t+1}
    vtrace_next = torch.zeros_like(values)
    vtrace_next[:, :-1] = vtrace_targets[:, 1:]
    vtrace_next[:, -1] = bootstrap_value
    vtrace_next = vtrace_next * not_done

    advantages = clipped_rhos * (rewards + gamma * vtrace_next - values)

    return vtrace_targets, advantages


# ============================================================================
# Training
# ============================================================================

def train(total_frames: int, num_workers: int, seed: int = 42, resume: Optional[str] = None):
    """Main training loop."""
    # Multiprocessing start method, should be called before any shared primitives
    mp.set_start_method("spawn")  # fork not working, workers stuck pulling SharedWeights, issues with WSL2 ?

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = IMPALAConfig(
        total_frames=total_frames,
        num_workers=num_workers,
        upgrade_levels = [0, 1, 2],
        game_steps_per_env = [1],
    )
    print(f"Config: {config}")

    # Create model
    model = ActorCritic().to(device)

    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Create shared weights
    shared_weights = SharedWeights(model)

    # Create communication primitives
    rollout_queue = Queue(maxsize=num_workers * 2)
    shutdown_event = Event()

    # Start workers
    workers = []
    for i in range(num_workers):
        args = i, rollout_queue, shared_weights, shutdown_event, config
        p = Process(target=worker_process, args=args, daemon=True)
        p.start()
        workers.append(p)

    print(f"Started {num_workers} workers")

    # LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-4)
    total_updates = total_frames // (config.rollout_length * num_workers)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, config.lr_start_factor, config.lr_end_factor, total_updates)

    # Tracking
    update_count = 0
    collected_frames = 0
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    start_time = time.time()

    print(f"\nStarting IMPALA training for {total_frames:,} frames...")
    print(f"Rollout length: {config.rollout_length}, Workers: {num_workers}")
    print(f"Frames per update: {config.rollout_length * num_workers:,}")

    try:
        while collected_frames < total_frames:
            rollouts = []
            frames_this_batch = 0
            target_frames = config.rollout_length * num_workers

            while frames_this_batch < target_frames:
                try:
                    rollout = rollout_queue.get(timeout=60.0)
                    rollouts.append(rollout)
                    frames_this_batch += len(rollout.rewards)

                    # Track episode statistics
                    for ret in rollout.episode_returns:
                        episode_returns.append(ret)
                    for length in rollout.episode_lengths:
                        episode_lengths.append(length)

                except Exception as e:
                    logger.warning(f"Timeout waiting for rollouts: {e}")
                    alive_workers = sum(1 for w in workers if w.is_alive())
                    if alive_workers == 0:
                        raise RuntimeError("All workers have died!")
                    continue

            collected_frames += frames_this_batch

            # Convert rollouts to tensors
            batch = prepare_batch(rollouts, device)

            # Compute V-trace targets with current policy
            model.eval()
            with torch.no_grad():
                # Get current policy log probs and values
                _, _, target_cmd_log_probs, target_angle_log_probs, _, current_values = model.get_action_and_value(
                    batch["observations"],
                    batch["commands"],
                    batch["angles"],
                )
                bootstrap_values = model.get_value(batch["next_observations"])

                # Combined log probs for importance sampling
                # For V-trace, we use the combined log prob
                target_log_probs = target_cmd_log_probs + target_angle_log_probs
                behavior_log_probs = batch["behavior_cmd_log_probs"] + batch["behavior_angle_log_probs"]

            vtrace_targets, advantages = compute_vtrace(
                behavior_log_probs=behavior_log_probs,
                target_log_probs=target_log_probs,
                rewards=batch["rewards"],
                values=batch["behavior_values"],  # Use behavior values for TD errors
                bootstrap_value=bootstrap_values,
                dones=batch["dones"],
                gamma=config.gamma,
                rho_bar=config.rho_bar,
                c_bar=config.c_bar,
            )

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten for mini-batch training
            B, T = batch["observations"].shape[:2]
            flat_obs = batch["observations"].reshape(B * T, -1)
            flat_commands = batch["commands"].reshape(B * T)
            flat_angles = batch["angles"].reshape(B * T, 2)
            flat_advantages = advantages.reshape(B * T)
            flat_vtrace_targets = vtrace_targets.reshape(B * T)
            flat_old_cmd_log_probs = batch["behavior_cmd_log_probs"].reshape(B * T)
            flat_old_angle_log_probs = batch["behavior_angle_log_probs"].reshape(B * T)

            # PPO-style mini-batch updates
            model.train()
            indices = np.arange(B * T)

            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            num_batches = 0

            for epoch in range(config.num_epochs):
                np.random.shuffle(indices)

                for start in range(0, B * T, config.mini_batch_size):
                    end = start + config.mini_batch_size
                    mb_indices = indices[start:end]

                    mb_obs = flat_obs[mb_indices]
                    mb_commands = flat_commands[mb_indices]
                    mb_angles = flat_angles[mb_indices]
                    mb_advantages = flat_advantages[mb_indices]
                    mb_vtrace_targets = flat_vtrace_targets[mb_indices]
                    mb_old_cmd_log_probs = flat_old_cmd_log_probs[mb_indices]
                    mb_old_angle_log_probs = flat_old_angle_log_probs[mb_indices]

                    # Forward pass
                    _, _, new_cmd_log_probs, new_angle_log_probs, entropy, new_values = model.get_action_and_value(
                        mb_obs, mb_commands, mb_angles
                    )

                    # === Command Policy Loss (standard PPO) ===
                    cmd_ratio = torch.exp(new_cmd_log_probs - mb_old_cmd_log_probs)
                    cmd_surr1 = cmd_ratio * mb_advantages
                    cmd_surr2 = torch.clamp(cmd_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * mb_advantages
                    cmd_policy_loss = -torch.min(cmd_surr1, cmd_surr2).mean()

                    # === Angle Policy Loss (masked by P(MOVE)) ===
                    # Option A: Only update when MOVE was selected (sparse but correct)
                    move_mask = (mb_commands == ACTION_MOVE).float()

                    # Option B: Weight by P(MOVE) for dense gradients (commented out alternative)
                    # features, cmd_logits, _ = model.forward(mb_obs)
                    # p_move = F.softmax(cmd_logits, dim=-1)[:, ACTION_MOVE]
                    # angle_weight = p_move

                    angle_ratio = torch.exp(new_angle_log_probs - mb_old_angle_log_probs)
                    angle_surr1 = angle_ratio * mb_advantages * move_mask
                    angle_surr2 = torch.clamp(angle_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * mb_advantages * move_mask

                    # Normalize by number of MOVE actions to avoid magnitude issues
                    num_moves = move_mask.sum().clamp(min=1.0)
                    angle_policy_loss = -torch.min(angle_surr1, angle_surr2).sum() / num_moves

                    # Combined policy loss
                    policy_loss = cmd_policy_loss + angle_policy_loss

                    # Value loss
                    value_loss = F.smooth_l1_loss(new_values, mb_vtrace_targets)

                    # Entropy bonus
                    entropy_loss = -entropy.mean()

                    # Total loss
                    loss = (
                        policy_loss
                        + config.value_coef * value_loss
                        + config.entropy_coef * entropy_loss
                    )

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                    total_loss += loss.item()
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    num_batches += 1

            # Update learning rate
            lr_scheduler.step()

            # Push updated weights to shared memory
            shared_weights.push(model)

            update_count += 1

            # Logging
            elapsed = time.time() - start_time
            fps = collected_frames / elapsed if elapsed > 0 else 0

            avg_return = np.mean(episode_returns) if episode_returns else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0  # noqa: F841
            win_rate = np.mean([r > 0 for r in episode_returns]) * 100 if episode_returns else 0.0

            # Weight staleness (how old are the weights workers are using)
            current_version = shared_weights.get_version()
            avg_staleness = np.mean([
                current_version - r.weight_version for r in rollouts
            ])

            print(
                f"Update {update_count:4d} | "
                f"Frames: {collected_frames:>10,} | "
                f"FPS: {fps:>6.0f} | "
                f"Return: {avg_return:>7.2f} | "
                f"Win: {win_rate:>5.1f}% | "
                f"Loss: {total_loss/num_batches:>6.3f} | "
                f"Entropy: {total_entropy/num_batches:>5.2f} | "
                f"Stale: {avg_staleness:>4.1f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        print("\nTraining complete!")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Shutdown workers
        print("Shutting down workers...")
        shutdown_event.set()

        # Give workers time to finish
        time.sleep(2.0)

        # Terminate any remaining workers
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join(timeout=1.0)

    # Save model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "collected_frames": collected_frames,
        "update_count": update_count,
        "config": config.__dict__,
    }

    model_path = model_dir / f"impala_v2_{timestamp}.pt"
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

    return str(model_path)


def prepare_batch(rollouts: list[Rollout], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert list of rollouts into batched tensors, stacking all rollouts (B, T, ...)"""
    return {
        "observations": torch.tensor(np.stack([r.observations for r in rollouts]), dtype=torch.float32, device=device),
        "commands": torch.tensor(np.stack([r.commands for r in rollouts]), dtype=torch.long, device=device),
        "angles": torch.tensor(np.stack([r.angles for r in rollouts]), dtype=torch.float32, device=device),
        "rewards": torch.tensor(np.stack([r.rewards for r in rollouts]), dtype=torch.float32, device=device),
        "dones": torch.tensor(np.stack([r.dones for r in rollouts]), dtype=torch.bool, device=device),
        "behavior_cmd_log_probs": torch.tensor(np.stack([r.behavior_cmd_log_probs for r in rollouts]), dtype=torch.float32, device=device),
        "behavior_angle_log_probs": torch.tensor(np.stack([r.behavior_angle_log_probs for r in rollouts]), dtype=torch.float32, device=device),
        "behavior_values": torch.tensor(np.stack([r.behavior_values for r in rollouts]), dtype=torch.float32, device=device),
        "next_observations": torch.tensor(np.stack([r.next_observation for r in rollouts]), dtype=torch.float32, device=device),
    }


# ============================================================================
# Evaluation
# ============================================================================

def eval_model(num_games: int = 10, model_path: Optional[str] = None):
    """Evaluate a trained model with hybrid action space."""

    device = torch.device("cpu")

    # Find latest model if not specified
    if model_path is None:
        model_files = glob.glob("artifacts/models/impala_v2_*.pt")
        if not model_files:
            print("No IMPALA models found!")
            return
        model_path = max(model_files, key=os.path.getctime)

    print(f"Loading model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model = ActorCritic().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wins = 0
    total_rewards = []
    total_lengths = []
    env = SC2GymEnv({"upgrade_level": [2], "game_steps_per_env": [1]})

    print(f"\nEvaluating for {num_games} games...")

    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # Use deterministic action for evaluation
                command, angle = model.get_deterministic_action(obs_tensor)

            # Create hybrid action dict
            action = {
                'command': command.squeeze(0).item(),
                'angle': angle.squeeze(0).numpy(),
            }

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_length += 1
            episode_reward += reward

        wins += episode_reward > 0
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Game {game + 1}: reward={episode_reward:.2f}, length={episode_length}, won={episode_reward > 0}")

    # Save replay of last game
    replay_data = env.game.clients[0].save_replay()
    replay_dir = PROJECT_ROOT / "artifacts" / "replays" / Path(model_path).stem
    replay_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    replay_path = replay_dir / f"{timestamp}.SC2Replay"
    replay_path.write_bytes(replay_data)
    print(f"Replay saved to {replay_path}")

    env.close()

    print(f"\n=== Results over {num_games} games ===")
    print(f"Win rate: {wins}/{num_games} ({100 * wins / num_games:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average length: {np.mean(total_lengths):.1f} ± {np.std(total_lengths):.1f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="IMPALA training with hybrid action space for Marine vs Zerglings")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument("--steps", type=int, default=1_000_000, help="Total training frames")
    train_parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    eval_parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")

    args = parser.parse_args()

    if args.command == "train":
        train(total_frames=args.steps, num_workers=args.num_workers, seed=args.seed, resume=args.resume)
    elif args.command == "eval":
        eval_model(num_games=args.games, model_path=args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
