"""Entity encoders for marine and enemy features.

These encoders transform raw observation features into dense embeddings
that can be processed by the temporal GRU and cross-attention layers.

Architecture:
    Marine features [hp_norm, cd_norm, can_attack] → MLP → marine_emb (d)
    Enemy features [hp_norm, sin, cos, dist] × N → shared MLP → enemy_embs (N, d)
"""

from torch import Tensor, nn

from ..config import (
    OBS_ENEMY_HP_OFFSET,
    OBS_ENEMY_START,
    OBS_MARINE_END,
    OBS_MARINE_START,
    OBS_TIME_LEFT_IDX,
    ModelConfig,
)


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]()


class EntityMLP(nn.Module):
    """Simple MLP for encoding entity features.

    Transforms raw features into a dense embedding with configurable
    number of layers and activation function.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        activation: str = "tanh",
        init_orthogonal: bool = True,
        init_gain: float = 1.41421356,
    ):
        super().__init__()

        layers = []
        in_size = input_size

        for i in range(num_layers):
            out_size = hidden_size if i < num_layers - 1 else output_size
            layers.append(nn.Linear(in_size, out_size))
            layers.append(get_activation(activation))
            in_size = out_size

        self.net = nn.Sequential(*layers)

        if init_orthogonal:
            self._init_weights(init_gain)

    def _init_weights(self, gain: float) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input features (..., input_size)

        Returns:
            Encoded features (..., output_size)
        """
        return self.net(x)


class MarineEncoder(nn.Module):
    """Encoder for marine-specific features.

    Input features:
        - hp_norm: Normalized health [0, 1]
        - cd_norm: Normalized weapon cooldown [0, 1]
        - can_attack: Binary flag (1 if weapon ready)

    Output:
        - marine_emb: Dense embedding (entity_embed_size,)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.mlp = EntityMLP(
            input_size=config.marine_obs_size,
            hidden_size=config.entity_hidden_size,
            output_size=config.entity_embed_size,
            num_layers=config.entity_num_layers,
            activation=config.entity_activation,
            init_orthogonal=config.init_orthogonal,
            init_gain=config.init_gain,
        )

    def forward(self, marine_obs: Tensor) -> Tensor:
        """Encode marine observation.

        Args:
            marine_obs: Marine features (B, marine_obs_size)

        Returns:
            Marine embedding (B, entity_embed_size)
        """
        return self.mlp(marine_obs)


class EnemyEncoder(nn.Module):
    """Encoder for enemy features with shared weights across enemies.

    Input features (per enemy):
        - hp_norm: Normalized health [0, 1]
        - sin_angle: Sine of relative angle [-1, 1]
        - cos_angle: Cosine of relative angle [-1, 1]
        - dist_norm: Normalized distance [0, 1]

    Output:
        - enemy_embs: Dense embeddings (N, entity_embed_size)

    The same MLP weights are shared across all enemies, allowing
    the model to generalize to variable numbers of enemies.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared MLP for all enemies
        self.mlp = EntityMLP(
            input_size=config.enemy_obs_size,
            hidden_size=config.entity_hidden_size,
            output_size=config.entity_embed_size,
            num_layers=config.entity_num_layers,
            activation=config.entity_activation,
            init_orthogonal=config.init_orthogonal,
            init_gain=config.init_gain,
        )

    def forward(self, enemy_obs: Tensor, enemy_mask: Tensor | None = None) -> Tensor:
        """Encode enemy observations.

        Args:
            enemy_obs: Enemy features (B, N, enemy_obs_size)
            enemy_mask: Optional mask where True = valid enemy (B, N)
                       Dead/invalid enemies are still encoded but should
                       be masked in downstream attention.

        Returns:
            Enemy embeddings (B, N, entity_embed_size)
        """
        batch_size, num_enemies, _ = enemy_obs.shape

        # Flatten for batch processing through shared MLP
        # Use reshape instead of view to handle non-contiguous tensors
        flat_obs = enemy_obs.reshape(batch_size * num_enemies, -1)
        flat_emb = self.mlp(flat_obs)

        # Reshape back to (B, N, embed_size)
        enemy_embs = flat_emb.reshape(batch_size, num_enemies, -1)

        # Zero out embeddings for invalid enemies (helps with attention)
        if enemy_mask is not None:
            # Expand mask to match embedding dimension
            mask_expanded = enemy_mask.unsqueeze(-1).float()
            enemy_embs = enemy_embs * mask_expanded

        return enemy_embs


class EntityEncoder(nn.Module):
    """Combined encoder for all entities (marine + enemies).

    This module handles the observation parsing and encoding for
    the entire observation space. It splits the flat observation
    into marine and enemy components, then encodes each.

    Observation format (from env.py):
        [0]     time_left       - Episode progress
        [1]     marine_hp       - Marine health fraction
        [2]     cd_binary       - Weapon on cooldown (0 or 1)
        [3]     cd_norm         - Weapon cooldown normalized
        [4:8]   z1_obs          - [hp, sin, cos, dist] for enemy 1
        [8:12]  z2_obs          - [hp, sin, cos, dist] for enemy 2
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.marine_encoder = MarineEncoder(config)
        self.enemy_encoder = EnemyEncoder(config)

    def parse_observation(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Parse flat observation into structured components.

        Args:
            obs: Flat observation (B, obs_size)

        Returns:
            Tuple of:
                - time_left: (B, 1)
                - marine_obs: (B, marine_obs_size) - [hp, cd_binary, cd_norm]
                - enemy_obs: (B, N, enemy_obs_size) - [hp, sin, cos, dist] per enemy
                - enemy_mask: (B, N) - True if enemy is valid (hp > 0)
        """
        batch_size = obs.shape[0]

        # Time left (using config constants)
        time_left = obs[:, OBS_TIME_LEFT_IDX : OBS_TIME_LEFT_IDX + 1]

        # Marine obs: hp, cd_binary, cd_norm (using config constants)
        marine_obs = obs[:, OBS_MARINE_START:OBS_MARINE_END]

        # Enemy obs: reshape the rest into (B, N, 4)
        enemy_flat = obs[:, OBS_ENEMY_START:]  # (B, N * enemy_obs_size)
        num_enemies = self.config.max_enemies
        enemy_obs = enemy_flat.view(batch_size, num_enemies, self.config.enemy_obs_size)

        # Create enemy mask based on hp (first feature per enemy)
        # Enemy is valid if hp > 0
        enemy_hp = enemy_obs[:, :, OBS_ENEMY_HP_OFFSET]
        enemy_mask = enemy_hp > 0

        return time_left, marine_obs, enemy_obs, enemy_mask

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode observation into entity embeddings.

        Args:
            obs: Flat observation (B, obs_size)

        Returns:
            Tuple of:
                - time_left: (B, 1)
                - marine_emb: (B, entity_embed_size)
                - enemy_embs: (B, N, entity_embed_size)
                - enemy_mask: (B, N) - True if enemy is valid
        """
        time_left, marine_obs, enemy_obs, enemy_mask = self.parse_observation(obs)

        marine_emb = self.marine_encoder(marine_obs)
        enemy_embs = self.enemy_encoder(enemy_obs, enemy_mask)

        return time_left, marine_emb, enemy_embs, enemy_mask

    def get_marine_obs(self, obs: Tensor) -> Tensor:
        """Extract just marine observation for skip connections.

        Args:
            obs: Flat observation (B, obs_size)

        Returns:
            Marine observation (B, marine_obs_size)
        """
        return obs[:, OBS_MARINE_START:OBS_MARINE_END]
