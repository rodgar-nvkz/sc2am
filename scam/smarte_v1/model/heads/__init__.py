"""Action, value, and auxiliary heads for the ActorCritic model.

Head Types:
    Action Heads (Hybrid Action Space):
        - ActionTypeHead: Discrete [MOVE, ATTACK, STOP]
        - MoveDirectionHead: Continuous [sin, cos] direction (conditional on MOVE)
        - AttackTargetHead: Pointer over N enemies (conditional on ATTACK)

    Value Heads:
        - CriticHead: Standard V(s) estimation
        - DualValueHead: Experimental win/lose conditional values

    Auxiliary Heads:
        - DamageAuxHead: Predict damage in next N steps
        - DistanceAuxHead: Predict distance to nearest enemy
        - CombinedAuxiliaryHead: Combined auxiliary predictions

Base Classes:
    - HeadOutput: Standardized output from combined action heads
    - HybridAction: Represents action type + conditional parameters
    - HeadLoss: Standardized loss output with metrics
    - ActionHead: Base class for action heads
    - ValueHead: Base class for value heads
    - AuxiliaryHead: Base class for auxiliary heads
"""

from .action_type import ActionTypeHead
from .attack_target import AttackTargetHead
from .auxiliary import CombinedAuxiliaryHead, DamageAuxHead, DistanceAuxHead
from .base import (
    ActionHead,
    AuxiliaryHead,
    HeadLoss,
    HeadOutput,
    HybridAction,
    ValueHead,
)
from .move_direction import MoveDirectionHead
from .value import CriticHead, DualValueHead

__all__ = [
    # Base classes and data structures
    "HeadOutput",
    "HybridAction",
    "HeadLoss",
    "ActionHead",
    "ValueHead",
    "AuxiliaryHead",
    # Action heads
    "ActionTypeHead",
    "MoveDirectionHead",
    "AttackTargetHead",
    # Value heads
    "CriticHead",
    "DualValueHead",
    # Auxiliary heads
    "DamageAuxHead",
    "DistanceAuxHead",
    "CombinedAuxiliaryHead",
]
