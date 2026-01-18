"""
Reward for Marine vs Zerglings.

Two-part reward:
1. Terminal: impala_v2 style (1 - enemy_hp_remaining_frac)
2. Gameplay: DPS momentum over fixed 16-step buckets
   - Idle penalty: zero DPS = nothing happening = bad (encourages chase)
   - Momentum: is the HP% trade ratio improving or getting worse?
   - Intensity weighting: faster combat = higher weight
"""

from dataclasses import dataclass, field
from typing import Protocol

MARINE_MAX_HP = 45.0
ZERGLING_MAX_HP = 35.0
NUM_ZERGLINGS = 2


class UnitLike(Protocol):
    health: float
    health_max: float


@dataclass
class RewardContext:
    marine: UnitLike | None
    zerglings: list[UnitLike]
    damage_dealt: float
    damage_taken: float
    current_step: int
    max_steps: int
    game_steps_per_env: int = 2

    @property
    def is_terminal(self) -> bool:
        marine_dead = self.marine is None
        zerglings_dead = len(self.zerglings) == 0
        timeout = self.current_step >= self.max_steps
        return marine_dead or zerglings_dead or timeout

    @property
    def enemy_hp_remaining_frac(self) -> float:
        if not self.zerglings:
            return 0.0
        return sum(z.health for z in self.zerglings) / (ZERGLING_MAX_HP * NUM_ZERGLINGS)

    @property
    def self_hp_remaining_frac(self) -> float:
        if not self.marine:
            return 0.0
        return self.marine.health / self.marine.health_max


class RewardStrategy:
    def reset(self) -> None:
        pass

    def compute(self, ctx: RewardContext) -> float:
        raise NotImplementedError


@dataclass
class SimpleReward(RewardStrategy):
    """
    Terminal: 1 - enemy_hp_remaining_frac (impala_v2 style)
    Gameplay: Trade ratio from exponential moving average (EMA)
              - idle penalty when nothing happening (encourages chase)
              - trade ratio: EMA of damage dealt vs taken (higher = better trades)
              - intensity weighting: faster combat = higher weight (20%/19% > 2%/1%)
    """

    idle_penalty: float = -0.001
    ratio_scale: float = 0.01
    intensity_scale: float = 1.0  # How much to weight by combat intensity

    # Exponential moving averages for damage rates
    ema_dealt: float = 0.0
    ema_taken: float = 0.0
    ema_alpha: float = 0.1  # EMA smoothing factor (higher = more weight to recent)

    def reset(self) -> None:
        self.ema_dealt = 0.0
        self.ema_taken = 0.0

    def compute(self, ctx: RewardContext) -> float:
        # Update EMAs with new damage values
        # EMA formula: ema = alpha * new_value + (1 - alpha) * ema
        self.ema_dealt = self.ema_alpha * ctx.damage_dealt + (1 - self.ema_alpha) * self.ema_dealt
        self.ema_taken = self.ema_alpha * ctx.damage_taken + (1 - self.ema_alpha) * self.ema_taken

        # Terminal reward
        if ctx.is_terminal:
            return ctx.self_hp_remaining_frac - ctx.enemy_hp_remaining_frac

        # Idle penalty: no combat activity in EMAs
        if self.ema_dealt < 0.001 and self.ema_taken < 0.001:
            return self.idle_penalty

        # Normalize EMAs to HP% scale
        total_enemy_hp = ZERGLING_MAX_HP * NUM_ZERGLINGS
        enemy_pct = self.ema_dealt / total_enemy_hp
        our_pct = self.ema_taken / MARINE_MAX_HP

        # Trade ratio: enemy_loss / our_loss (higher = better for us)
        # Centered around 1.0: ratio > 1 means good trade, < 1 means bad
        eps = 0.001
        ratio = enemy_pct / (our_pct + eps)
        centered_ratio = ratio - 1.0  # positive = good, negative = bad

        # Intensity: total HP% in current EMA (faster combat = more weight)
        # This makes "20% us / 19% them" worth more than "2% us / 1% them"
        intensity = enemy_pct + our_pct
        weighted_ratio = centered_ratio * (1.0 + intensity * self.intensity_scale)

        return weighted_ratio * self.ratio_scale


def create_reward_strategy(name: str = "simple", **kwargs) -> RewardStrategy:
    if name == "simple":
        return SimpleReward(**kwargs)
    raise ValueError(f"Unknown strategy: {name}")
