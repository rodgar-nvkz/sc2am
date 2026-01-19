import dataclasses

from .model.config import ModelConfig


@dataclasses.dataclass
class IMPALAConfig:
    """IMPALA training configuration - simplified for 1 game = 1 episode."""

    # Model configuration
    model: ModelConfig

    # Batch settings
    num_workers: int = 8
    episodes_per_batch: int = 16  # Collect this many complete episodes before training

    # Training settings
    total_episodes: int = 10_000  # Train for this many episodes
    num_epochs: int = 4  # Training passes per batch

    # V-trace / GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.99  # GAE λ for variance reduction (0=TD(0), 1=MC)
    c_bar: float = 1.0  # Truncation for trace coefficients
    rho_bar: float = 1.0  # Truncation for importance weights

    # PPO-style clipping
    clip_epsilon: float = 0.2

    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 40.0

    # Auxiliary task coefficient
    # The auxiliary task forces the encoder to represent observation features
    # (enemy angles, distances) that are critical for correct action selection.
    # This prevents encoder collapse where policy gradients cancel across episodes.
    # Start with 0.5 (equal to value_coef), tune down if it dominates policy learning.
    aux_coef: float = 0.5

    # Optimizer
    lr: float = 5e-3
    lr_eps: float = 1e-4
    lr_start_factor: float = 1.0
    lr_end_factor: float = 0.1

    # Environment
    upgrade_levels: list = dataclasses.field(default_factory=list)
