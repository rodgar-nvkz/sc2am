"""LSTM-based ActorCritic model for sequential decision making.

Architecture:
    obs (obs_size) → VectorEncoder (MLP) → LSTM → [ActorHead, CriticHead]

The LSTM enables the agent to maintain memory across timesteps,
which is crucial for learning chase → attack → kite sequences.

Action space (40 discrete actions):
    - 0-35: MOVE in direction (angle = i * 10°)
    - 36: ATTACK_Z1
    - 37: ATTACK_Z2
    - 38: STOP
    - 39: SKIP (no-op)
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .encoders import VectorEncoder
from .heads import CriticHead, DiscreteActionHead, HeadLoss, HeadOutput


@dataclass
class ActorCriticOutput:
    """Complete output from ActorCritic forward pass."""

    # Action head output
    action: HeadOutput

    # Value estimate
    value: Tensor

    # New hidden state (for next step)
    hidden: tuple[Tensor, Tensor]

    @property
    def entropy(self) -> Tensor:
        """Action entropy."""
        return self.action.entropy

    @property
    def log_prob(self) -> Tensor:
        """Action log probability."""
        return self.action.log_prob


class ActorCritic(nn.Module):
    """LSTM-based actor-critic for discrete action space.

    Architecture:
        obs → VectorEncoder → features
                                  ↓
                               LSTM(features, hidden) → lstm_out, new_hidden
                                  ↓
              [lstm_out, obs] → ActorHead → action distribution
                    ↓
              [lstm_out, obs] → CriticHead → value

    The skip connection (concatenating raw obs) preserves precise angle information
    for movement decisions.
    """

    def __init__(self, config: ModelConfig):
        """Initialize ActorCritic model.

        Args:
            config: Model configuration defining architecture.
        """
        super().__init__()
        self.config = config

        # Encoder: obs → features
        self.encoder = VectorEncoder(config)

        # LSTM: features → lstm_out
        self.lstm = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
        )

        # Initialize LSTM weights
        if config.init_orthogonal:
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.orthogonal_(param, gain=config.init_gain)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param, gain=config.init_gain)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
                    # Set forget gate bias to 1 for better gradient flow
                    hidden_size = config.lstm_hidden_size
                    param.data[hidden_size:2*hidden_size].fill_(1.0)

        # Heads: lstm_out (+ skip) → action/value
        self.action_head = DiscreteActionHead(config)
        self.value_head = CriticHead(config)

    def get_initial_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        """Get zero-initialized hidden state for LSTM.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h_0, c_0) each with shape (num_layers, batch_size, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device

        h_0 = torch.zeros(
            self.config.lstm_num_layers,
            batch_size,
            self.config.lstm_hidden_size,
            device=device,
        )
        c_0 = torch.zeros(
            self.config.lstm_num_layers,
            batch_size,
            self.config.lstm_hidden_size,
            device=device,
        )
        return (h_0, c_0)

    def forward(
        self,
        obs: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
        action: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> ActorCriticOutput:
        """Forward pass for single timestep.

        Args:
            obs: Observations (B, obs_size)
            hidden: LSTM hidden state tuple (h, c). If None, uses zeros.
            action: Optional action to evaluate (B,). If None, samples new.
            action_mask: Optional boolean mask where True = valid action (B, num_actions)

        Returns:
            ActorCriticOutput with action output, value, and new hidden state
        """
        batch_size = obs.shape[0]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs.device)

        # Encode observation
        features = self.encoder(obs)  # (B, embed_size)

        # Add sequence dimension for LSTM
        features = features.unsqueeze(1)  # (B, 1, embed_size)

        # LSTM forward
        lstm_out, new_hidden = self.lstm(features, hidden)  # (B, 1, hidden_size)

        # Remove sequence dimension
        lstm_out = lstm_out.squeeze(1)  # (B, hidden_size)

        # Action head
        action_output = self.action_head(
            features=lstm_out,
            raw_obs=obs,
            action=action,
            mask=action_mask,
        )

        # Value head
        value = self.value_head(features=lstm_out, raw_obs=obs)

        return ActorCriticOutput(
            action=action_output,
            value=value,
            hidden=new_hidden,
        )

    def forward_sequence(
        self,
        obs_seq: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
        actions: Tensor | None = None,
        action_masks: Tensor | None = None,
    ) -> ActorCriticOutput:
        """Forward pass for a sequence of observations (e.g., full episode).

        This is more efficient than calling forward() in a loop because
        it processes the entire sequence through the LSTM at once.

        Args:
            obs_seq: Observation sequence (B, T, obs_size)
            hidden: Initial LSTM hidden state. If None, uses zeros.
            actions: Optional actions to evaluate (B, T). If None, samples new.
            action_masks: Optional masks (B, T, num_actions)

        Returns:
            ActorCriticOutput with:
                - action.action: (B, T)
                - action.log_prob: (B, T)
                - action.entropy: (B, T)
                - value: (B, T)
                - hidden: final hidden state
        """
        batch_size, seq_len, _ = obs_seq.shape

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs_seq.device)

        # Encode all observations
        obs_flat = obs_seq.view(batch_size * seq_len, -1)  # (B*T, obs_size)
        features_flat = self.encoder(obs_flat)  # (B*T, embed_size)
        features = features_flat.view(batch_size, seq_len, -1)  # (B, T, embed_size)

        # LSTM forward through entire sequence
        lstm_out, new_hidden = self.lstm(features, hidden)  # (B, T, hidden_size)

        # Flatten for heads
        lstm_out_flat = lstm_out.reshape(batch_size * seq_len, -1)  # (B*T, hidden_size)
        obs_flat = obs_seq.reshape(batch_size * seq_len, -1)  # (B*T, obs_size)

        # Prepare actions and masks
        actions_flat = None
        if actions is not None:
            actions_flat = actions.reshape(batch_size * seq_len)  # (B*T,)

        masks_flat = None
        if action_masks is not None:
            masks_flat = action_masks.reshape(batch_size * seq_len, -1)  # (B*T, num_actions)

        # Action head
        action_output = self.action_head(
            features=lstm_out_flat,
            raw_obs=obs_flat,
            action=actions_flat,
            mask=masks_flat,
        )

        # Value head
        values_flat = self.value_head(features=lstm_out_flat, raw_obs=obs_flat)

        # Reshape outputs back to (B, T)
        action_out_reshaped = HeadOutput(
            action=action_output.action.view(batch_size, seq_len),
            log_prob=action_output.log_prob.view(batch_size, seq_len),
            entropy=action_output.entropy.view(batch_size, seq_len),
            distribution=None,  # Can't easily reshape distribution
        )

        return ActorCriticOutput(
            action=action_out_reshaped,
            value=values_flat.view(batch_size, seq_len),
            hidden=new_hidden,
        )

    def get_value(self, obs: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """Get value estimate only.

        Args:
            obs: Observations (B, obs_size)
            hidden: LSTM hidden state. If None, uses zeros.

        Returns:
            Value estimates (B,)
        """
        output = self.forward(obs, hidden)
        return output.value

    def get_deterministic_action(
        self,
        obs: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Get deterministic action (argmax) for evaluation.

        Args:
            obs: Observations (B, obs_size)
            hidden: LSTM hidden state. If None, uses zeros.
            action_mask: Optional boolean mask where True = valid action

        Returns:
            Tuple of:
            - action: (B,) discrete action indices
            - new_hidden: updated LSTM hidden state
        """
        batch_size = obs.shape[0]

        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs.device)

        # Encode and LSTM
        features = self.encoder(obs).unsqueeze(1)
        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out.squeeze(1)

        # Deterministic action
        action = self.action_head.get_deterministic_action(
            features=lstm_out,
            raw_obs=obs,
            mask=action_mask,
        )

        return action, new_hidden

    def compute_losses(
        self,
        output: ActorCriticOutput,
        old_log_prob: Tensor,
        advantages: Tensor,
        vtrace_targets: Tensor,
        clip_epsilon: float,
    ) -> dict[str, HeadLoss]:
        """Compute losses for all heads.

        Args:
            output: Forward pass output (with flattened tensors)
            old_log_prob: Behavior policy log probs (B*T,) or (N,)
            advantages: Advantage estimates (B*T,) or (N,)
            vtrace_targets: V-trace value targets (B*T,) or (N,)
            clip_epsilon: PPO clipping parameter

        Returns:
            Dictionary of HeadLoss for each head
        """
        # Flatten output tensors if they have sequence dimension
        new_log_prob = output.action.log_prob
        values = output.value

        if new_log_prob.dim() > 1:
            new_log_prob = new_log_prob.reshape(-1)
            values = values.reshape(-1)

        action_loss = self.action_head.compute_loss(
            new_log_prob=new_log_prob,
            old_log_prob=old_log_prob,
            advantages=advantages,
            clip_epsilon=clip_epsilon,
        )

        value_loss = self.value_head.compute_loss(
            values=values,
            targets=vtrace_targets,
        )

        return {
            "action": action_loss,
            "value": value_loss,
        }
