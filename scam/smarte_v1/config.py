import dataclasses
from typing import Literal

from .model.config import ModelConfig


@dataclasses.dataclass
class IMPALAConfig:
    """IMPALA training configuration - simplified for 1 game = 1 episode."""

    # Model configuration
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    # Batch settings
    num_workers: int = 8
    episodes_per_batch: int = 16  # Collect this many complete episodes before training
    max_episode_steps: int = 1024  # Safety limit (game usually ends in ~200-600)

    # Training settings
    total_episodes: int = 10_000  # Train for this many episodes
    num_epochs: int = 4  # Training passes per batch

    # V-trace / GAE parameters
    gamma: float = 0.99
    c_bar: float = 1.0  # Truncation for trace coefficients
    rho_bar: float = 1.0  # Truncation for importance weights

    # Worker settings
    weight_sync_interval: int = 5  # Sync weights every N episodes (reduces lock contention)

    # PPO-style clipping
    clip_epsilon: float = 0.2

    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 40.0

    # Entropy annealing: high early entropy encourages exploration (critical for chase learning)
    # Start high to explore movement options, decay as policy converges
    entropy_coef_start: float = 0.05  # 5x default, encourages exploration early
    entropy_coef_end: float = 0.005  # Decay to this value
    entropy_anneal: bool = True  # Enable entropy annealing

    # Optimizer
    lr: float = 5e-4
    lr_eps: float = 1e-4
    lr_start_factor: float = 1.0
    lr_end_factor: float = 0.2

    # Environment
    upgrade_levels: list = dataclasses.field(default_factory=list)

    # Reward strategy
    reward_strategy: str = "simple"
    reward_params: dict = dataclasses.field(default_factory=lambda: {})

    def get_entropy_coef(self, progress: float) -> float:
        """Get entropy coefficient based on training progress (0 to 1).

        Higher entropy early encourages exploration, which is critical for
        discovering that movement toward enemies leads to combat rewards.
        """
        if not self.entropy_anneal:
            return self.entropy_coef

        # Linear decay from start to end
        return self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * progress
