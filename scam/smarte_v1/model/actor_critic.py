"""Entity-attention based ActorCritic model for SC2 marine micro.

Architecture:
    Observation → EntityEncoder → [marine_emb, enemy_embs]
                        ↓
    TemporalEncoder (GRU) → [h_marine, h_enemies]
                        ↓
    CrossAttention (marine→enemies) → context, attn_weights
                        ↓
    SharedBackbone output: [h_marine; context]
                        ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
ActionTypeHead    MoveDirectionHead    AttackTargetHead
    ↓                   ↓                   ↓
 [MOVE,ATTACK,STOP]  [sin,cos]         enemy_idx
    ↓                   ↓                   ↓
    └─────── HybridAction ─────────────────┘
                        ↓
              ValueHead → V(s)
              AuxiliaryHeads → damage_pred, distance_pred

Key Features:
    1. Entity-based encoding for variable number of enemies
    2. GRU temporal encoding for memory across steps
    3. Cross-attention for interpretable enemy targeting
    4. Conditional action heads (only active when relevant)
    5. Auxiliary tasks for richer training signal
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .attention import CrossAttention
from .config import ACTION_ATTACK, ACTION_MOVE, ACTION_STOP, ModelConfig
from .encoders import EntityEncoder, TemporalEncoder
from .heads import (
    ActionTypeHead,
    AttackTargetHead,
    CombinedAuxiliaryHead,
    CriticHead,
    HeadLoss,
    HybridAction,
    MoveDirectionHead,
)


@dataclass
class ActorCriticOutput:
    """Complete output from ActorCritic forward pass.

    Attributes:
        action: HybridAction with type, direction, and target
        log_prob: Combined log probability of the action
        entropy: Combined entropy of the action distribution
        value: Value estimate V(s)
        hidden: New hidden state for next step (h_marine, h_enemies)
        attn_weights: Attention weights over enemies (useful for visualization)

        Component log probs (always stored for training):
            action_type_log_prob: Log prob of action type
            move_direction_log_prob: Log prob of direction (computed for all samples,
                                     but only contributes to loss for MOVE actions)
            attack_target_log_prob: Log prob of target (computed for all samples,
                                    but only contributes to loss for ATTACK actions)

        Auxiliary predictions (optional):
            aux_damage: Predicted damage in next N steps
            aux_distance: Predicted distance to nearest enemy
    """

    action: HybridAction
    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    hidden: tuple[Tensor, Tensor]
    attn_weights: Tensor

    # Component log probs (always stored, masking applied during loss computation)
    action_type_log_prob: Tensor
    move_direction_log_prob: Tensor  # Always computed, masked during training
    attack_target_log_prob: Tensor  # Always computed, masked during training

    # Auxiliary predictions
    aux_damage: Tensor | None = None
    aux_distance: Tensor | None = None


class ActorCritic(nn.Module):
    """Entity-attention based actor-critic for hybrid action space.

    This model processes observations through:
        1. Entity encoders (separate for marine and enemies)
        2. Temporal GRU (maintains memory across steps)
        3. Cross-attention (marine attends to enemies)
        4. Multiple output heads (action type, direction, target, value, auxiliary)

    The architecture supports:
        - Variable number of enemies (via attention masking)
        - Hybrid action space (discrete type + conditional continuous/discrete params)
        - Temporal context (GRU hidden state across steps)
        - Auxiliary tasks for richer training signal
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # === Entity Encoder ===
        self.entity_encoder = EntityEncoder(config)

        # === Temporal Encoder ===
        self.temporal_encoder = TemporalEncoder(config)

        # === Cross-Attention ===
        self.cross_attention = CrossAttention(config)

        # === Action Heads ===
        self.action_type_head = ActionTypeHead(config)
        self.move_direction_head = MoveDirectionHead(config)
        self.attack_target_head = AttackTargetHead(config)

        # === Value Head ===
        self.value_head = CriticHead(config)

        # === Auxiliary Heads ===
        if config.use_auxiliary_tasks:
            self.auxiliary_head = CombinedAuxiliaryHead(config)
        else:
            self.auxiliary_head = None

    def get_initial_hidden(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Get zero-initialized hidden state for temporal GRU.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h_marine, h_enemies):
                - h_marine: (num_layers, B, gru_hidden_size)
                - h_enemies: (num_layers, B, max_enemies, gru_hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device

        return self.temporal_encoder.get_initial_hidden(
            batch_size=batch_size,
            num_enemies=self.config.max_enemies,
            device=device,
        )

    def _compute_action_mask(
        self,
        marine_obs: Tensor,
        enemy_mask: Tensor,
        range_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute action type mask based on game state.

        Args:
            marine_obs: Marine observation (B, marine_obs_size)
                       [hp_norm, cd_binary, cd_norm]
            enemy_mask: Valid enemy mask (B, N)
            range_mask: In-range enemy mask (B, N) or None

        Returns:
            Action mask (B, 3) where True = valid action
        """
        # Extract can_attack from marine_obs (cd_binary is index 1, 0 means can attack)
        can_attack = marine_obs[:, 1] < 0.5  # cd_binary == 0 means weapon ready

        # Check if any enemy is in range
        if range_mask is not None:
            has_valid_target = (enemy_mask & range_mask).any(dim=-1)
        else:
            # If no range mask provided, assume all alive enemies are targetable
            has_valid_target = enemy_mask.any(dim=-1)

        # Create action mask
        return ActionTypeHead.create_action_mask(can_attack, has_valid_target)

    def forward(
        self,
        obs: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
        action: HybridAction | None = None,
        action_mask: Tensor | None = None,
        range_mask: Tensor | None = None,
    ) -> ActorCriticOutput:
        """Forward pass for single timestep.

        Args:
            obs: Flat observation (B, obs_size)
            hidden: Temporal hidden state (h_marine, h_enemies). If None, uses zeros.
            action: Optional action to evaluate. If None, samples new action.
            action_mask: Optional action type mask (B, 3). If None, computed from obs.
            range_mask: Optional attack range mask (B, N). If None, uses enemy_mask.

        Returns:
            ActorCriticOutput with action, value, hidden state, etc.
        """
        batch_size = obs.shape[0]

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs.device)

        # === Entity Encoding ===
        time_left, marine_emb, enemy_embs, enemy_mask = self.entity_encoder(obs)
        marine_obs = self.entity_encoder.get_marine_obs(obs)

        # === Temporal Encoding ===
        h_marine, h_enemies, new_hidden = self.temporal_encoder(
            marine_emb, enemy_embs, hidden, enemy_mask
        )

        # === Cross-Attention ===
        context, attn_weights, attn_logits = self.cross_attention(
            h_marine, h_enemies, enemy_mask
        )

        # === Backbone Output ===
        # Include time_left if configured (helps value function understand episode progress)
        if self.config.use_time_feature:
            backbone_out = torch.cat([h_marine, context, time_left], dim=-1)
        else:
            backbone_out = torch.cat([h_marine, context], dim=-1)

        # === Compute Action Mask ===
        if action_mask is None:
            action_mask = self._compute_action_mask(marine_obs, enemy_mask, range_mask)

        # === Action Type Head ===
        action_type_input = action.action_type if action is not None else None
        action_type, action_type_log_prob, action_type_entropy, _ = self.action_type_head(
            features=backbone_out,
            marine_obs=marine_obs,
            action_type=action_type_input,
            action_mask=action_mask,
        )

        # === Conditional Action Heads ===
        # Move direction (only for MOVE actions)
        is_move = action_type == ACTION_MOVE
        move_direction_input = action.move_direction if action is not None else None
        move_direction, move_log_prob, move_entropy, _ = self.move_direction_head(
            features=backbone_out,
            marine_obs=marine_obs,
            direction=move_direction_input,
        )

        # Attack target (only for ATTACK actions)
        is_attack = action_type == ACTION_ATTACK
        attack_target_input = action.attack_target if action is not None else None

        # Use range_mask for attack targeting if provided
        target_range_mask = range_mask if range_mask is not None else enemy_mask
        attack_target, attack_log_prob, attack_entropy, _ = self.attack_target_head(
            attn_logits=attn_logits,
            enemy_mask=enemy_mask,
            range_mask=target_range_mask,
            attack_target=attack_target_input,
        )

        # === Combine Log Probs and Entropy ===
        # Only include conditional log probs for the selected action type
        combined_log_prob = action_type_log_prob.clone()
        combined_entropy = action_type_entropy.clone()

        # Add move direction log prob for MOVE actions
        combined_log_prob = torch.where(
            is_move,
            combined_log_prob + move_log_prob,
            combined_log_prob,
        )
        combined_entropy = torch.where(
            is_move,
            combined_entropy + move_entropy,
            combined_entropy,
        )

        # Add attack target log prob for ATTACK actions
        combined_log_prob = torch.where(
            is_attack,
            combined_log_prob + attack_log_prob,
            combined_log_prob,
        )
        combined_entropy = torch.where(
            is_attack,
            combined_entropy + attack_entropy,
            combined_entropy,
        )

        # === Value Head ===
        value = self.value_head(features=backbone_out, marine_obs=marine_obs)

        # === Auxiliary Heads ===
        aux_damage = None
        aux_distance = None
        if self.auxiliary_head is not None:
            aux_preds = self.auxiliary_head(features=backbone_out, marine_obs=marine_obs)
            aux_damage = aux_preds["damage"]
            aux_distance = aux_preds["distance"]

        # === Construct Output ===
        hybrid_action = HybridAction(
            action_type=action_type,
            move_direction=move_direction,
            attack_target=attack_target,
        )

        return ActorCriticOutput(
            action=hybrid_action,
            log_prob=combined_log_prob,
            entropy=combined_entropy,
            value=value,
            hidden=new_hidden,
            attn_weights=attn_weights,
            action_type_log_prob=action_type_log_prob,
            move_direction_log_prob=move_log_prob,  # Always stored, masked during loss
            attack_target_log_prob=attack_log_prob,  # Always stored, masked during loss
            aux_damage=aux_damage,
            aux_distance=aux_distance,
        )

    def forward_sequence(
        self,
        obs_seq: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
        actions: dict[str, Tensor] | None = None,
        action_masks: Tensor | None = None,
        range_masks: Tensor | None = None,
    ) -> ActorCriticOutput:
        """Forward pass for a sequence of observations.

        More efficient than step-by-step for training on complete episodes.

        Args:
            obs_seq: Observation sequence (B, T, obs_size)
            hidden: Initial hidden state. If None, uses zeros.
            actions: Optional dict with:
                - "action_type": (B, T)
                - "move_direction": (B, T, 2)
                - "attack_target": (B, T)
            action_masks: Optional action masks (B, T, 3)
            range_masks: Optional range masks (B, T, N)

        Returns:
            ActorCriticOutput with sequence outputs (B, T, ...)
        """
        batch_size, seq_len, _ = obs_seq.shape

        # Initialize hidden state
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs_seq.device)

        # Process step by step (could be optimized with sequence GRU later)
        outputs = []
        current_hidden = hidden

        for t in range(seq_len):
            obs_t = obs_seq[:, t]

            # Get action for this timestep if provided
            action_t = None
            if actions is not None:
                action_t = HybridAction(
                    action_type=actions["action_type"][:, t],
                    move_direction=actions["move_direction"][:, t],
                    attack_target=actions["attack_target"][:, t],
                )

            action_mask_t = action_masks[:, t] if action_masks is not None else None
            range_mask_t = range_masks[:, t] if range_masks is not None else None

            output_t = self.forward(
                obs=obs_t,
                hidden=current_hidden,
                action=action_t,
                action_mask=action_mask_t,
                range_mask=range_mask_t,
            )

            outputs.append(output_t)
            current_hidden = output_t.hidden

        # Stack outputs
        return self._stack_outputs(outputs, current_hidden)

    def _stack_outputs(
        self,
        outputs: list[ActorCriticOutput],
        final_hidden: tuple[Tensor, Tensor],
    ) -> ActorCriticOutput:
        """Stack list of single-step outputs into sequence output."""
        # Stack action components
        action_type = torch.stack([o.action.action_type for o in outputs], dim=1)
        move_direction = torch.stack([o.action.move_direction for o in outputs], dim=1)
        attack_target = torch.stack([o.action.attack_target for o in outputs], dim=1)

        # Stack other outputs
        log_prob = torch.stack([o.log_prob for o in outputs], dim=1)
        entropy = torch.stack([o.entropy for o in outputs], dim=1)
        value = torch.stack([o.value for o in outputs], dim=1)
        attn_weights = torch.stack([o.attn_weights for o in outputs], dim=1)

        action_type_log_prob = torch.stack([o.action_type_log_prob for o in outputs], dim=1)
        move_direction_log_prob = torch.stack([o.move_direction_log_prob for o in outputs], dim=1)
        attack_target_log_prob = torch.stack([o.attack_target_log_prob for o in outputs], dim=1)

        # Handle optional fields - filter out None values before stacking
        aux_damage = None
        aux_distance = None
        if outputs[0].aux_damage is not None:
            aux_damage_list = [o.aux_damage for o in outputs if o.aux_damage is not None]
            aux_distance_list = [o.aux_distance for o in outputs if o.aux_distance is not None]
            if aux_damage_list:
                aux_damage = torch.stack(aux_damage_list, dim=1)
            if aux_distance_list:
                aux_distance = torch.stack(aux_distance_list, dim=1)

        return ActorCriticOutput(
            action=HybridAction(
                action_type=action_type,
                move_direction=move_direction,
                attack_target=attack_target,
            ),
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            hidden=final_hidden,
            attn_weights=attn_weights,
            action_type_log_prob=action_type_log_prob,
            move_direction_log_prob=move_direction_log_prob,
            attack_target_log_prob=attack_target_log_prob,
            aux_damage=aux_damage,
            aux_distance=aux_distance,
        )

    def get_value(
        self,
        obs: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Get value estimate only (for bootstrapping).

        Args:
            obs: Observations (B, obs_size)
            hidden: Hidden state. If None, uses zeros.

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
        range_mask: Tensor | None = None,
    ) -> tuple[HybridAction, tuple[Tensor, Tensor]]:
        """Get deterministic action (argmax/mean) for evaluation.

        Args:
            obs: Observations (B, obs_size)
            hidden: Hidden state. If None, uses zeros.
            action_mask: Optional action type mask (B, 3)
            range_mask: Optional range mask (B, N)

        Returns:
            Tuple of (HybridAction, new_hidden)
        """
        batch_size = obs.shape[0]

        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, obs.device)

        # Encode
        time_left, marine_emb, enemy_embs, enemy_mask = self.entity_encoder(obs)
        marine_obs = self.entity_encoder.get_marine_obs(obs)

        # Temporal
        h_marine, h_enemies, new_hidden = self.temporal_encoder(
            marine_emb, enemy_embs, hidden, enemy_mask
        )

        # Attention
        context, attn_weights, attn_logits = self.cross_attention(
            h_marine, h_enemies, enemy_mask
        )

        # Backbone (include time_left if configured)
        if self.config.use_time_feature:
            backbone_out = torch.cat([h_marine, context, time_left], dim=-1)
        else:
            backbone_out = torch.cat([h_marine, context], dim=-1)

        # Action mask
        if action_mask is None:
            action_mask = self._compute_action_mask(marine_obs, enemy_mask, range_mask)

        # Deterministic action type
        action_type = self.action_type_head.get_deterministic_action(
            features=backbone_out,
            marine_obs=marine_obs,
            action_mask=action_mask,
        )

        # Deterministic move direction
        move_direction = self.move_direction_head.get_deterministic_action(
            features=backbone_out,
            marine_obs=marine_obs,
        )

        # Deterministic attack target
        target_range_mask = range_mask if range_mask is not None else enemy_mask
        attack_target = self.attack_target_head.get_deterministic_action(
            attn_logits=attn_logits,
            enemy_mask=enemy_mask,
            range_mask=target_range_mask,
        )

        hybrid_action = HybridAction(
            action_type=action_type,
            move_direction=move_direction,
            attack_target=attack_target,
        )

        return hybrid_action, new_hidden

    def compute_losses(
        self,
        output: ActorCriticOutput,
        old_log_probs: dict[str, Tensor],
        advantages: Tensor,
        value_targets: Tensor,
        clip_epsilon: float,
        aux_targets: dict[str, Tensor] | None = None,
    ) -> dict[str, HeadLoss]:
        """Compute all losses for training.

        Args:
            output: Forward pass output
            old_log_probs: Dict with:
                - "combined": Combined log probs from behavior policy (B, T) or (N,)
                - "action_type": Action type log probs (B, T) or (N,)
                - "move_direction": Move direction log probs (B, T) or (N,)
                - "attack_target": Attack target log probs (B, T) or (N,)
            advantages: Advantage estimates (B, T) or (N,)
            value_targets: Value targets (B, T) or (N,)
            clip_epsilon: PPO clipping parameter
            aux_targets: Optional dict with auxiliary targets

        Returns:
            Dict of HeadLoss for each component
        """
        # Flatten if needed
        action_type = output.action.action_type
        if action_type.dim() > 1:
            action_type = action_type.reshape(-1)
            action_type_log_prob = output.action_type_log_prob.reshape(-1)
            value = output.value.reshape(-1)
            advantages = advantages.reshape(-1)
            value_targets = value_targets.reshape(-1)
        else:
            action_type_log_prob = output.action_type_log_prob
            value = output.value

        # Create action type masks
        is_move = action_type == ACTION_MOVE
        is_attack = action_type == ACTION_ATTACK

        losses = {}

        # === Action Type Loss ===
        losses["action_type"] = self.action_type_head.compute_loss(
            new_log_prob=action_type_log_prob,
            old_log_prob=old_log_probs["action_type"].reshape(-1),
            advantages=advantages,
            clip_epsilon=clip_epsilon,
        )

        # === Move Direction Loss (masked to MOVE actions) ===
        # Use stored log probs directly - no need to compute via subtraction
        move_direction_log_prob = output.move_direction_log_prob
        if move_direction_log_prob.dim() > 1:
            move_direction_log_prob = move_direction_log_prob.reshape(-1)

        if is_move.any() and "move_direction" in old_log_probs:
            losses["move_direction"] = self.move_direction_head.compute_loss(
                new_log_prob=move_direction_log_prob,
                old_log_prob=old_log_probs["move_direction"].reshape(-1),
                advantages=advantages,
                clip_epsilon=clip_epsilon,
                mask=is_move,
            )
        else:
            losses["move_direction"] = HeadLoss(
                loss=torch.tensor(0.0, device=value.device),
                metrics={"loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0, "num_move_actions": 0},
            )

        # === Attack Target Loss (masked to ATTACK actions) ===
        # Use stored log probs directly - no need to compute via subtraction
        attack_target_log_prob = output.attack_target_log_prob
        if attack_target_log_prob.dim() > 1:
            attack_target_log_prob = attack_target_log_prob.reshape(-1)

        if is_attack.any() and "attack_target" in old_log_probs:
            losses["attack_target"] = self.attack_target_head.compute_loss(
                new_log_prob=attack_target_log_prob,
                old_log_prob=old_log_probs["attack_target"].reshape(-1),
                advantages=advantages,
                clip_epsilon=clip_epsilon,
                mask=is_attack,
            )
        else:
            losses["attack_target"] = HeadLoss(
                loss=torch.tensor(0.0, device=value.device),
                metrics={"loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0, "num_attack_actions": 0},
            )

        # === Value Loss ===
        losses["value"] = self.value_head.compute_loss(
            values=value,
            targets=value_targets,
        )

        # === Auxiliary Losses ===
        if self.auxiliary_head is not None and aux_targets is not None:
            if output.aux_damage is not None and output.aux_distance is not None:
                aux_preds: dict[str, Tensor] = {
                    "damage": output.aux_damage.reshape(-1),
                    "distance": output.aux_distance.reshape(-1),
                }
                losses["auxiliary"] = self.auxiliary_head.compute_loss(
                    predictions=aux_preds,
                    targets={k: v.reshape(-1) for k, v in aux_targets.items()},
                )

        return losses

    def to_env_action(self, action: HybridAction, batch_idx: int = 0) -> dict:
        """Convert HybridAction to environment action format.

        Args:
            action: HybridAction from model output
            batch_idx: Index in batch to extract

        Returns:
            Dict with "command" and "angle" for environment
        """
        # Detach all tensors before extracting values (consistent handling)
        action_type = action.action_type[batch_idx].detach().item()
        move_direction = action.move_direction[batch_idx].detach()
        attack_target = action.attack_target[batch_idx].detach().item()

        # Convert direction to numpy
        angle = move_direction.cpu().numpy()

        if action_type == ACTION_MOVE:
            return {
                "command": ACTION_MOVE,
                "angle": angle,
            }
        elif action_type == ACTION_ATTACK:
            # Map attack_target (0 or 1) to command (1 or 2)
            return {
                "command": 1 + attack_target,  # ATTACK_Z1=1, ATTACK_Z2=2
                "angle": angle,  # Not used but required
            }
        else:  # STOP
            return {
                "command": ACTION_STOP,
                "angle": angle,  # Not used but required
            }
