"""Angle head for continuous movement direction with auxiliary prediction.

AlphaZero-style: predicts movement angle from observations only,
without conditioning on the selected command. The angle represents
"if we move, where should we go?" - a property of the state, not
of the command choice.

Auxiliary Prediction Task:
The auxiliary head forces the encoder to represent observation features
(enemy angles, distances) that are critical for correct action selection.
This prevents encoder collapse where policy gradients cancel across episodes.

Masking is applied at loss computation time, not during forward pass.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, distributions, nn

from ..config import ModelConfig
from .base import ActionHead, HeadLoss, HeadOutput


class AngleHead(ActionHead):
    """Continuous action head for movement angle with auxiliary prediction.

    Outputs a direction vector (sin, cos) for movement.
    Predicts from observations only (no command conditioning).

    The distribution is a Normal over (sin, cos) with learnable log_std.
    Outputs are normalized to the unit circle.

    Auxiliary Task:
    An auxiliary head predicts selected observation features from the
    hidden representation. This supervised loss doesn't cancel across
    episodes (unlike policy gradient), forcing the encoder to maintain
    observation-dependent representations.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Split network into encoder + output layer to expose hidden state
        # The encoder is what the auxiliary task forces to learn good representations
        self.encoder = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
        )
        self.output_layer = nn.Linear(config.head_hidden_size, 2)  # sin, cos

        # Learnable log std (shared across all states)
        self.log_std = nn.Parameter(torch.tensor([config.angle_init_log_std, config.angle_init_log_std]))

        # Auxiliary prediction head - predicts observation features from hidden state
        # This forces encoder to represent angle/distance information
        self.aux_enabled = config.aux_enabled
        if self.aux_enabled:
            self.aux_head = nn.Sequential(
                nn.Linear(config.head_hidden_size, config.aux_hidden_size),
                nn.Tanh(),
                nn.Linear(config.aux_hidden_size, config.aux_target_size),
            )
            self.aux_target_indices = config.aux_target_indices

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        if not self.config.init_orthogonal:
            return

        # Initialize encoder
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        # Initialize output layer
        nn.init.orthogonal_(self.output_layer.weight, gain=self.config.init_gain)
        nn.init.constant_(self.output_layer.bias, 0.0)

        # Initialize auxiliary head if enabled
        if self.aux_enabled:
            for module in self.aux_head:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: Tensor, action: Tensor | None = None) -> HeadOutput:
        """Forward pass: produce angle distribution and sample/evaluate.

        Args:
            obs: Raw observation (B, obs_size)
            action: Optional angle to evaluate (B, 2). If None, samples new action.

        Returns:
            HeadOutput with continuous angle action (normalized sin, cos)
        """
        # Encode observation to hidden state
        h = self.encoder(obs)

        # Output raw direction - no normalization!
        # Normalization was causing gradient issues (KL stayed ~0 despite non-zero gradients)
        # The environment will normalize for movement; here we just need consistent gradients
        mean = self.output_layer(h)

        std = self.log_std.exp()
        dist = distributions.Normal(mean, std)

        if action is None:
            # Normalizing after sampling breaks the log_prob gradient relationship
            action = dist.sample()

        # Log prob: sum over sin/cos dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return HeadOutput(action=action, log_prob=log_prob, entropy=entropy, distribution=dist)

    def compute_aux_loss(self, obs: Tensor) -> Tensor:
        """Compute auxiliary prediction loss.

        The auxiliary head predicts selected observation features from the
        hidden representation. This supervised loss forces the encoder to
        represent observation-dependent information (enemy angles, distances).

        Unlike policy gradient, this loss doesn't cancel across episodes:
        - Predicting "enemy at angle θ₁" vs "enemy at angle θ₂" are distinct targets
        - Each gradient updates different directions in weight space
        - The encoder MUST differentiate observations to minimize this loss

        Args:
            obs: Raw observation (B, obs_size)

        Returns:
            Scalar MSE loss for auxiliary prediction
        """
        if not self.aux_enabled:
            return torch.tensor(0.0, device=obs.device)

        # Get hidden representation (same as used for policy)
        h = self.encoder(obs)

        # Predict auxiliary targets
        aux_pred = self.aux_head(h)

        # Extract ground truth targets from observation
        # These are the features we want the encoder to represent
        aux_targets = obs[:, self.aux_target_indices]

        # MSE loss - supervised, doesn't cancel!
        return F.mse_loss(aux_pred, aux_targets)

    def get_deterministic_action(self, obs: Tensor) -> Tensor:
        """Get deterministic action (mean of distribution) for evaluation.

        Args:
            obs: Raw observation (B, obs_size)

        Returns:
            Raw angle (sin, cos) tensor (B, 2)
        """
        h = self.encoder(obs)
        return self.output_layer(h)

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
        mask: Tensor,
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss for angle head.

        Only computes loss for steps where MOVE command was selected.

        Args:
            new_log_prob: Log prob from current policy (B,)
            old_log_prob: Log prob from behavior policy (B,)
            advantages: Advantage estimates (B,)
            clip_epsilon: PPO clipping parameter
            mask: Float mask where 1.0 = MOVE command (B,).

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        ratio = torch.exp(new_log_prob - old_log_prob)

        # Masked loss: only compute for MOVE commands
        surr1 = ratio * advantages * mask
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages * mask
        num_moves = mask.sum().clamp(min=1.0)
        loss = -torch.min(surr1, surr2).sum() / num_moves

        # Metrics only for masked steps
        with torch.no_grad():
            if mask.sum() > 0:
                masked_old = old_log_prob[mask.bool()]
                masked_new = new_log_prob[mask.bool()]
                masked_ratio = ratio[mask.bool()]
                approx_kl = (masked_old - masked_new).mean().item()
                clip_fraction = ((masked_ratio - 1.0).abs() > clip_epsilon).float().mean().item()
            else:
                approx_kl = 0.0
                clip_fraction = 0.0

        metrics = {"loss": loss.item(), "approx_kl": approx_kl, "clip_fraction": clip_fraction}
        return HeadLoss(loss=loss, metrics=metrics)
