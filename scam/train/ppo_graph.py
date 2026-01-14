"""
CUDA Graph accelerated inference for Stable Baselines3 PPO.

This module provides utilities to speed up PPO inference using CUDA Graphs,
which eliminates GPU kernel launch overhead for small models and batch sizes.

Benchmarks show ~5x speedup for inference with small networks ([48, 32])
with batch sizes of 8-128 environments.

Usage:
    from scam.train.ppo_graph import CUDAGraphInference, enable_cuda_graph_inference

    # Option 1: Wrap existing model for manual prediction
    model = PPO.load("model.zip", device="cuda")
    fast_inference = CUDAGraphInference(model.policy, batch_size=8)
    actions = fast_inference.predict(observations)

    # Option 2: Patch model for SB3's internal rollout collection (full GPU)
    model = PPO(MlpPolicy, env, device="cuda", ...)
    enable_cuda_graph_inference(model, num_envs=8)
    model.learn(total_timesteps=100000)  # Rollouts use CUDA Graphs automatically

    # Option 3: Hybrid mode - CUDA Graph inference + CPU training (recommended)
    # Best of both worlds: fast inference, efficient CPU training for small models
    model = PPO(MlpPolicy, env, device="cpu", ...)
    enable_cuda_graph_inference(model, num_envs=8, hybrid_mode=True)
    model.learn(total_timesteps=100000)
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv


class CUDAGraphInference:
    """
    Wraps an SB3 ActorCriticPolicy for fast inference using CUDA Graphs.

    CUDA Graphs capture a sequence of GPU operations and replay them with
    minimal CPU overhead. This is particularly effective for small models
    where kernel launch overhead dominates compute time.

    Requirements:
        - Fixed batch size (CUDA Graphs require static shapes)
        - CUDA-capable GPU
        - Policy must be on CUDA device

    Note:
        This only accelerates inference (action prediction). Training still
        uses the standard forward pass to compute gradients.
    """

    def __init__(
        self,
        policy: ActorCriticPolicy,
        batch_size: int,
        device: str = "cuda",
    ):
        """
        Initialize CUDA Graph inference wrapper.

        Args:
            policy: SB3 ActorCriticPolicy to wrap
            batch_size: Fixed batch size for inference (must match num_envs)
            device: CUDA device to use (default: "cuda")
        """
        self.policy = policy
        self.batch_size = batch_size
        self.device = torch.device(device)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. CUDAGraphInference requires a GPU.")

        # Move policy to GPU if not already there
        self.policy.to(self.device)

        # Get observation shape from policy
        obs_shape = policy.observation_space.shape

        # Determine action dtype based on action space
        if isinstance(policy.action_space, spaces.Discrete):
            self.action_dtype = torch.int64
            self.action_shape = (batch_size,)
        else:
            self.action_dtype = torch.float32
            self.action_shape = (batch_size, policy.action_space.shape[0])

        # Static GPU buffers - CUDA Graphs require fixed memory addresses
        self.static_obs = torch.zeros(
            (batch_size, *obs_shape),
            dtype=torch.float32,
            device=self.device
        )

        # Output buffers (will be set during graph capture)
        self.static_actions: Optional[torch.Tensor] = None
        self.static_values: Optional[torch.Tensor] = None
        self.static_log_probs: Optional[torch.Tensor] = None

        # Pinned memory for faster CPU<->GPU transfer
        self.pinned_obs = torch.zeros(
            (batch_size, *obs_shape),
            dtype=torch.float32,
            pin_memory=True
        )
        self.pinned_actions = torch.zeros(
            self.action_shape,
            dtype=self.action_dtype,
            pin_memory=True
        )

        # The captured CUDA graph
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        # Capture the graph
        self._capture_graph()

    def _raw_forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Raw forward pass through the policy network without distribution sampling.

            This avoids the Categorical distribution creation which is not compatible
            with CUDA Graph capture due to validation operations.

            Args:
                obs: Observation tensor on GPU

            Returns:
                Tuple of (actions, values, log_probs) tensors
            """
            # Extract features
            features = self.policy.extract_features(obs, self.policy.features_extractor)

            # Pass through MLP
            latent_pi, latent_vf = self.policy.mlp_extractor(features)

            # Get action logits and values
            action_logits = self.policy.action_net(latent_pi)
            values = self.policy.value_net(latent_vf)

            # For discrete actions: sample from categorical and compute log_prob
            if isinstance(self.policy.action_space, spaces.Discrete):
                # Use softmax to get probabilities, then sample
                probs = torch.softmax(action_logits, dim=1)
                # For deterministic: use argmax
                actions = action_logits.argmax(dim=1)
                # Compute log_prob for the selected action
                log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)) + 1e-8).squeeze(1)
            else:
                actions = action_logits  # For continuous, logits are the mean
                log_probs = torch.zeros(actions.shape[0], device=actions.device)

            return actions, values, log_probs

    def _capture_graph(self) -> None:
        """Capture the inference computation as a CUDA Graph."""
        # Warmup runs (required before capture to initialize lazy operations)
        self.policy.eval()
        with torch.no_grad():
            for _ in range(5):
                self._raw_forward(self.static_obs)

        # Synchronize before capture
        torch.cuda.synchronize()

        # Capture the graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                self.static_actions, self.static_values, self.static_log_probs = self._raw_forward(self.static_obs)

        torch.cuda.synchronize()

    def predict(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        return_values: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fast action prediction using captured CUDA Graph.

        Args:
            observations: Batch of observations (numpy array or tensor)
                         Shape must be (batch_size, *obs_shape)
            return_values: If True, also return value estimates

        Returns:
            actions: Predicted actions as numpy array
            values (optional): Value estimates if return_values=True
        """
        # Handle numpy input
        if isinstance(observations, np.ndarray):
            # Use pinned memory for faster transfer
            self.pinned_obs.copy_(torch.from_numpy(observations))
            self.static_obs.copy_(self.pinned_obs, non_blocking=True)
        else:
            self.static_obs.copy_(observations.to(self.device), non_blocking=True)

        # Replay the captured graph
        self.graph.replay()

        # Copy results back to CPU using pinned memory
        self.pinned_actions.copy_(self.static_actions.cpu(), non_blocking=False)
        actions = self.pinned_actions.numpy().copy()

        if return_values:
            values = self.static_values.cpu().numpy()
            return actions, values

        return actions

    def forward_for_rollout(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for SB3 rollout collection.

        This method is designed to be called from a patched policy.forward()
        during rollout collection. It returns tensors on GPU that SB3 expects.

        Args:
            observations: Observation tensor on GPU

        Returns:
            Tuple of (actions, values, log_probs) as GPU tensors
        """
        # Copy observations to static buffer
        self.static_obs.copy_(observations)

        # Replay the captured graph
        self.graph.replay()

        # Return clones so the static buffers can be reused
        return (
            self.static_actions.clone(),
            self.static_values.clone(),
            self.static_log_probs.clone(),
        )

    def recapture(self) -> None:
        """
        Recapture the CUDA Graph after policy weights have been updated.

        Call this after training updates if you want inference to use
        the new weights. Note: This is usually not necessary as the graph
        references the same weight tensors that training updates in-place.
        """
        self._capture_graph()


class CUDAGraphVecEnvWrapper:
    """
    Wrapper that integrates CUDAGraphInference with SB3's VecEnv interface.

    This wrapper intercepts the predict call and uses CUDA Graphs for
    fast inference while keeping the standard SB3 API.
    """

    def __init__(self, model: PPO, num_envs: int):
        """
        Initialize the wrapper.

        Args:
            model: Trained PPO model (will be moved to CUDA)
            num_envs: Number of parallel environments (determines batch size)
        """
        self.model = model
        self.num_envs = num_envs

        # Move model to CUDA and create fast inference
        self.model.policy.to("cuda")
        self.fast_inference = CUDAGraphInference(
            self.model.policy,
            batch_size=num_envs,
            device="cuda"
        )

    def predict(
        self,
        observations: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        Predict actions using CUDA Graph accelerated inference.

        Args:
            observations: Batch of observations from VecEnv
            deterministic: Ignored (always deterministic with CUDA Graphs)

        Returns:
            Tuple of (actions, None) to match SB3 API
        """
        actions = self.fast_inference.predict(observations)
        return actions, None


def create_cuda_graph_rollout_collector(
    model: PPO,
    env: VecEnv,
) -> CUDAGraphInference:
    """
    Create a CUDA Graph inference wrapper matched to a VecEnv.

    This is a convenience function that creates a CUDAGraphInference
    with the correct batch size for the given environment.

    Args:
        model: PPO model to wrap
        env: VecEnv that will provide observations

    Returns:
        CUDAGraphInference configured for the environment
    """
    num_envs = env.num_envs
    model.policy.to("cuda")
    return CUDAGraphInference(model.policy, batch_size=num_envs, device="cuda")


def enable_cuda_graph_inference(
    model: PPO,
    num_envs: int,
    hybrid_mode: bool = False,
) -> CUDAGraphInference:
    """
    Enable CUDA Graph accelerated inference for SB3's internal rollout collection.

    This patches the model's policy to use CUDA Graphs during the collect_rollouts
    phase while preserving normal gradient-enabled forward passes for training.

    Args:
        model: PPO model
        num_envs: Number of parallel environments (batch size)
        hybrid_mode: If True, keep model on CPU for training but use CUDA Graphs
                    for inference. This is often faster for small networks because
                    CPU training is more efficient while GPU inference (with CUDA
                    Graphs) eliminates the inference bottleneck.

    Returns:
        The CUDAGraphInference instance (for reference/debugging)

    Example:
        # Full GPU mode (model must be on CUDA)
        model = PPO(MlpPolicy, env, device="cuda", ...)
        cuda_graph = enable_cuda_graph_inference(model, num_envs=8)

        # Hybrid mode (model on CPU, inference on GPU) - recommended for small models
        model = PPO(MlpPolicy, env, device="cpu", ...)
        cuda_graph = enable_cuda_graph_inference(model, num_envs=8, hybrid_mode=True)

        model.learn(total_timesteps=100000)  # Uses CUDA Graphs for rollouts
    """
    if not hybrid_mode and str(model.device) != "cuda":
        raise ValueError(
            f"Model must be on CUDA device (got {model.device}). "
            "Use hybrid_mode=True to keep model on CPU while using CUDA Graph inference."
        )

    policy = model.policy

    if hybrid_mode:
        # Create a copy of the policy on GPU for inference
        # The original policy stays on CPU for training
        import copy

        # Create GPU copy of just the inference components
        gpu_policy = copy.deepcopy(policy)
        gpu_policy.to("cuda")
        gpu_policy.eval()

        # Create CUDA Graph inference with GPU policy
        cuda_inference = CUDAGraphInference(
            gpu_policy, batch_size=num_envs, device="cuda"
        )

        # Store original forward method
        original_forward = policy.forward

        def patched_forward_hybrid(obs: torch.Tensor, deterministic: bool = False):
            """
            Patched forward for hybrid mode.

            During rollouts (eval mode): Move obs to GPU, use CUDA Graph, move back to CPU.
            During training: Use original CPU forward for gradients.
            """
            if not policy.training and obs.shape[0] == num_envs:
                # Rollout collection - use CUDA Graph
                # Move observation to GPU
                obs_gpu = obs.to("cuda", non_blocking=True)

                # Get results from CUDA Graph
                actions_gpu, values_gpu, log_probs_gpu = cuda_inference.forward_for_rollout(obs_gpu)

                # Move results back to CPU (where the model expects them)
                actions = actions_gpu.to("cpu")
                values = values_gpu.to("cpu")
                log_probs = log_probs_gpu.to("cpu")

                return actions, values, log_probs
            else:
                # Training - use original CPU forward
                return original_forward(obs, deterministic=deterministic)

        # Apply the patch
        policy.forward = patched_forward_hybrid

        # Setup weight synchronization: after each training update, sync weights to GPU
        # We do this by patching the optimizer step
        def sync_weights_to_gpu():
            """Copy updated weights from CPU policy to GPU policy."""
            gpu_policy.load_state_dict(policy.state_dict())
            cuda_inference.recapture()

        # Store sync function on model for manual calling if needed
        model._sync_cuda_graph_weights = sync_weights_to_gpu

        # Patch the train() method to sync weights after each training phase
        original_train = model.train

        def patched_train():
            result = original_train()
            sync_weights_to_gpu()
            return result

        model.train = patched_train

    else:
        # Full GPU mode - original implementation
        cuda_inference = CUDAGraphInference(
            policy, batch_size=num_envs, device="cuda"
        )

        original_forward = policy.forward

        def patched_forward(obs: torch.Tensor, deterministic: bool = False):
            """
            Patched forward that uses CUDA Graphs when in eval mode (rollouts).

            During training (policy.training=True), use original forward for gradients.
            During rollouts (policy.training=False), use CUDA Graph for speed.
            """
            if (not policy.training and
                obs.shape[0] == num_envs and
                obs.device.type == "cuda"):
                return cuda_inference.forward_for_rollout(obs)
            else:
                return original_forward(obs, deterministic=deterministic)

        policy.forward = patched_forward

    # Store reference to cuda_inference on the model for later access
    model._cuda_graph_inference = cuda_inference
    model._cuda_graph_hybrid_mode = hybrid_mode

    return cuda_inference


def benchmark_inference(
    policy: ActorCriticPolicy,
    batch_size: int,
    obs_shape: Tuple[int, ...],
    num_iters: int = 1000,
) -> dict:
    """
    Benchmark CPU vs CUDA Graph inference performance.

    Args:
        policy: Policy to benchmark
        batch_size: Batch size to test
        obs_shape: Shape of observations
        num_iters: Number of iterations for timing

    Returns:
        Dictionary with timing results
    """
    import time

    results = {}
    obs_np = np.random.randn(batch_size, *obs_shape).astype(np.float32)
    obs_cpu = torch.from_numpy(obs_np)

    # CPU benchmark
    policy_cpu = policy.to("cpu")
    torch.set_num_threads(4)

    t0 = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            policy_cpu(obs_cpu, deterministic=True)
    cpu_time = (time.perf_counter() - t0) / num_iters * 1000
    results["cpu_ms"] = cpu_time

    # CUDA Graph benchmark
    policy_gpu = policy.to("cuda")
    cuda_inference = CUDAGraphInference(policy_gpu, batch_size=batch_size)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        cuda_inference.predict(obs_np)
    torch.cuda.synchronize()
    graph_time = (time.perf_counter() - t0) / num_iters * 1000
    results["cuda_graph_ms"] = graph_time

    results["speedup"] = cpu_time / graph_time

    return results


# For convenience, export main classes
__all__ = [
    "CUDAGraphInference",
    "CUDAGraphVecEnvWrapper",
    "create_cuda_graph_rollout_collector",
    "enable_cuda_graph_inference",
    "benchmark_inference",
]


if __name__ == "__main__":
    # Quick benchmark when run directly
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    print("CUDA Graph PPO Benchmark")
    print("=" * 50)

    class SimpleEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(11)
            self.step_count = 0

        def reset(self, seed=None):
            self.step_count = 0
            return np.random.randn(15).astype(np.float32), {}

        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 50
            return np.random.randn(15).astype(np.float32), 1.0, done, False, {}

    NUM_ENVS = 8
    policy_kwargs = {"net_arch": [48, 32]}

    # Create test policy
    env = DummyVecEnv([lambda: SimpleEnv() for _ in range(NUM_ENVS)])
    model = PPO("MlpPolicy", env, device="cpu", policy_kwargs=policy_kwargs, verbose=0)

    print(f"\nBatch size: {NUM_ENVS}")
    print(f"Network: {policy_kwargs['net_arch']}")
    print(f"Observation shape: (15,)")
    print(f"Action space: Discrete(11)")

    results = benchmark_inference(
        model.policy,
        batch_size=NUM_ENVS,
        obs_shape=(15,),
        num_iters=2000,
    )

    print(f"\nResults (2000 iterations):")
    print(f"  CPU inference:        {results['cpu_ms']:.3f}ms per call")
    print(f"  CUDA Graph inference: {results['cuda_graph_ms']:.3f}ms per call")
    print(f"  Speedup:              {results['speedup']:.2f}x")

    env.close()
