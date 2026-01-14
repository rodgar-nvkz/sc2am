import numpy as np
import torch.nn.functional as F
from torch import Tensor, cat, distributions, nn, tensor


class ActorCritic(nn.Module):
    """
    Autoregressive actor-critic for hybrid discrete-continuous action space.

    Action space example:
    - Discrete command: STAY, MOVE, ATTACK_Z1, ATTACK_Z2 (4 options)
    - Continuous angle: (sin, cos) for movement direction (only used when command=MOVE)

    The angle head is conditioned on the discrete command via embedding.
    This allows the network to learn command-specific angle distributions.

    P(action | obs) = P(command | obs) * P(angle | obs, command)
    """

    COMMAND_EMBED_SHAPE = 16

    def __init__(self, obs_size: int, act_size: int):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )

        # Discrete command head
        self.command_head = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, act_size),
        )

        # Command embedding (for conditioning angle head)
        self.command_embed = nn.Embedding(act_size, 16)

        # Angle head, conditioned on command via embedding. Outputs mean of (sin, cos) for movement direction
        self.angle_head = nn.Sequential(
            nn.Linear(64 + 16, 32), nn.Tanh(),  # features + command embedding
            nn.Linear(32, 2),  # sin, cos mean
        )

        # Learnable log std for angle (shared across all states)
        self.angle_logstd = nn.Parameter(tensor([-0.5, -0.5]))

        # Critic head (only depends on observation, not action)
        self.critic = nn.Sequential(
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def _init_weights(self):
        """Initialize network weights (orthogonal like PPO)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0.0)
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)

        # Smaller init for output layers
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)  # type: ignore
        nn.init.orthogonal_(self.angle_head[-1].weight, gain=0.01)  # type: ignore
        nn.init.orthogonal_(self.command_head[-1].weight, gain=0.01)  # type: ignore

    def forward(self, obs: Tensor):
        """Forward pass returning features and command logits."""
        features = self.shared(obs)
        cmd_logits = self.command_head(features)
        value = self.critic(features)
        return features, cmd_logits, value

    def get_value(self, obs: Tensor) -> Tensor:
        """Get value estimate only."""
        features = self.shared(obs)
        return self.critic(features).squeeze(-1)

    def get_angle_distribution(self, features: Tensor, command: Tensor):
        """Get angle distribution conditioned on command. Returns normal distribution over (sin, cos) angle encoding"""
        cmd_embed = self.command_embed(command)  # (B, 16)
        angle_input = cat([features, cmd_embed], dim=-1)  # (B, 64+16)
        angle_mean = self.angle_head(angle_input)  # (B, 2)

        # Normalize to unit circle for stability (optional but helps)
        angle_mean = F.normalize(angle_mean, dim=-1)

        angle_std = self.angle_logstd.exp()
        return distributions.Normal(angle_mean, angle_std)

    def get_action_and_value(self, obs: Tensor, command: Tensor | None = None, angle: Tensor | None = None):
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
        cmd_dist = distributions.Categorical(logits=cmd_logits)

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

    def get_deterministic_action(self, obs: Tensor):
        """Get deterministic action for evaluation (argmax command, mean angle)."""
        features, cmd_logits, _ = self.forward(obs)

        # Deterministic command (argmax)
        command = cmd_logits.argmax(dim=-1)

        # Deterministic angle (mean of distribution)
        angle_dist = self.get_angle_distribution(features, command)
        angle = F.normalize(angle_dist.mean, dim=-1)

        return command, angle
