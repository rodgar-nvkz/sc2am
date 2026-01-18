"""Action and value heads for the ActorCritic model."""

from smarte.impala_v2.model.heads.angle import AngleHead
from smarte.impala_v2.model.heads.base import ActionHead, HeadLoss, HeadOutput
from smarte.impala_v2.model.heads.command import CommandHead
from smarte.impala_v2.model.heads.value import CriticHead

__all__ = [
    "ActionHead",
    "AngleHead",
    "CommandHead",
    "CriticHead",
    "HeadLoss",
    "HeadOutput",
]
