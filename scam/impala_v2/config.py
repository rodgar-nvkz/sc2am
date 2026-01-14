import dataclasses


@dataclasses.dataclass
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
    c_bar: float = 1.0    # Truncation for trace coefficients
    rho_bar: float = 1.0  # Truncation for importance weights

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
    upgrade_levels: list = dataclasses.field(default_factory=list)
    game_steps_per_env: list = dataclasses.field(default_factory=list)
