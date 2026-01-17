"""Discrete action type head for MOVE/ATTACK/STOP selection.

This head outputs a categorical distribution over action types:
    - MOVE (0): Move in a direction (requires move_direction head)
    - ATTACK (1): Attack an enemy (requires attack_target head)
    - STOP (2): Hold position

Action Masking:
    - MOVE: Always available
    - ATTACK: Masked if weapon on cooldown OR no enemies in attack range
    - STOP: Always available

The masking is critical for proper credit assignment - the policy should
never be penalized for not choosing an unavailable action.
"""

import torch
from torch import Tensor, distributions, nn

from ..config import ACTION_ATTACK, ACTION_MOVE, ACTION_STOP, ModelConfig
from .base import ActionHead, HeadLoss


class ActionTypeHead(ActionHead):
    """Categorical head for discrete action type selection.

    Input: Backbone output (+ optional skip connection)
    Output: Distribution over [MOVE, ATTACK, STOP]

    The action type determines which conditional head is active:
        - If MOVE: move_direction head is used
        - If ATTACK: attack_target head is used
        - If STOP: no additional parameters needed
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # MLP: backbone_out (+ skip) -> action type logits
        self.net = nn.Sequential(
            nn.Linear(config.head_input_size, config.head_hidden_size),
            nn.Tanh(),
            nn.Linear(config.head_hidden_size, config.num_action_types),
        )

        if config.init_orthogonal:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=self.config.init_gain)
                nn.init.constant_(module.bias, 0.0)

        # Small init for output layer (policy head)
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=self.config.policy_init_gain)

    def forward(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
        action_type: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, distributions.Categorical]:
        """Forward pass: produce action type distribution.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)
            action_type: Optional action type to evaluate (B,). If None, samples new.
            action_mask: Boolean mask where True = valid action (B, num_action_types)
                        Default: [True, True, True] (all actions valid)

        Returns:
            Tuple of:
                - action_type: Sampled or provided action type (B,)
                - log_prob: Log probability of action type (B,)
                - entropy: Entropy of the distribution (B,)
                - distribution: Categorical distribution object
        """
        # Build input with optional skip connection
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        # Get logits
        logits = self.net(x)  # (B, num_action_types)

        # Apply action mask
        if action_mask is not None:
            # Mask invalid actions by setting logits to -inf
            logits = logits.masked_fill(~action_mask, float("-inf"))

        # Create distribution
        dist = distributions.Categorical(logits=logits)

        # Sample or evaluate
        if action_type is None:
            action_type = dist.sample()

        log_prob = dist.log_prob(action_type)
        entropy = dist.entropy()

        return action_type, log_prob, entropy, dist

    def get_deterministic_action(
        self,
        features: Tensor,
        marine_obs: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> Tensor:
        """Get deterministic action (argmax) for evaluation.

        Args:
            features: Backbone output (B, backbone_output_size)
            marine_obs: Raw marine observation for skip connection (B, marine_obs_size)
            action_mask: Boolean mask where True = valid action (B, num_action_types)

        Returns:
            Action type indices (B,)
        """
        if self.config.use_skip_connections and marine_obs is not None:
            x = torch.cat([features, marine_obs], dim=-1)
        else:
            x = features

        logits = self.net(x)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return logits.argmax(dim=-1)

    def compute_loss(
        self,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantages: Tensor,
        clip_epsilon: float,
        mask: Tensor | None = None,
    ) -> HeadLoss:
        """Compute PPO-clipped policy loss for action type.

        Args:
            new_log_prob: Log prob from current policy (B,)
            old_log_prob: Log prob from behavior policy (B,)
            advantages: Advantage estimates (B,)
            clip_epsilon: PPO clipping parameter
            mask: Optional mask where True = include in loss (B,)

        Returns:
            HeadLoss with loss tensor and metrics dict
        """
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss_per_sample = -torch.min(surr1, surr2)

        if mask is not None:
            mask_float = mask.float()
            num_valid = mask_float.sum().clamp(min=1.0)
            loss = (loss_per_sample * mask_float).sum() / num_valid
        else:
            loss = loss_per_sample.mean()

        # Compute metrics
        with torch.no_grad():
            approx_kl = (old_log_prob - new_log_prob).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

        return HeadLoss(
            loss=loss,
            metrics={
                "loss": loss.item(),
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
            },
        )

    @staticmethod
    def create_action_mask(
        can_attack: Tensor,
        has_valid_target: Tensor,
    ) -> Tensor:
        """Create action mask from game state.

        Args:
            can_attack: Whether weapon is ready (B,) boolean
            has_valid_target: Whether any enemy is in range (B,) boolean

        Returns:
            Action mask (B, 3) where True = valid action
        """
        batch_size = can_attack.shape[0]
        device = can_attack.device

        mask = torch.ones(batch_size, 3, dtype=torch.bool, device=device)

        # MOVE (0): Always available
        mask[:, ACTION_MOVE] = True

        # ATTACK (1): Only if weapon ready AND valid target exists
        mask[:, ACTION_ATTACK] = can_attack & has_valid_target

        # STOP (2): Always available
        mask[:, ACTION_STOP] = True

        return mask
