"""Entity and temporal encoders for the ActorCritic model.

Entity Encoders:
    - MarineEncoder: MLP for marine features
    - EnemyEncoder: Shared MLP for variable number of enemies
    - EntityEncoder: Combined encoder that parses observations

Temporal Encoders:
    - TemporalEncoder: GRU-based encoder maintaining state across steps
    - TemporalEncoderSequence: Efficient sequence processing for training
"""

from .entity import EnemyEncoder, EntityEncoder, EntityMLP, MarineEncoder
from .temporal import TemporalEncoder, TemporalEncoderSequence
from .vector import VectorEncoder

__all__ = [
    # Entity encoders
    "EntityEncoder",
    "EntityMLP",
    "MarineEncoder",
    "EnemyEncoder",
    # Temporal encoders
    "TemporalEncoder",
    "TemporalEncoderSequence",
    # Legacy (kept for compatibility)
    "VectorEncoder",
]
