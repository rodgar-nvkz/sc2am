"""Comprehensive tests for masking behavior in ActorCritic model.

These tests verify that:
1. Action masking correctly prevents invalid command selection
2. Angle loss is properly masked when command != MOVE
3. Entropy is properly masked for non-MOVE commands
4. Log probabilities are properly masked for importance sampling
5. Gradients only flow through relevant paths

This is crucial for correct training behavior in the AlphaZero-style
parallel prediction architecture.
"""

import pytest
import torch

from smarte.smarte_v3.env import SC2GymEnv
from smarte.smarte_v3.model.actor_critic import ActorCritic
from smarte.smarte_v3.model.config import ModelConfig
from smarte.smarte_v3.model.heads import AngleHead

# Import action constants from environment class
ACTION_MOVE = SC2GymEnv.ACTION_MOVE
ACTION_ATTACK_Z1 = SC2GymEnv.ACTION_ATTACK_Z1
ACTION_ATTACK_Z2 = SC2GymEnv.ACTION_ATTACK_Z2


@pytest.fixture
def config() -> ModelConfig:
    """Create a test configuration."""
    return ModelConfig(obs_size=11, num_commands=3, move_action_id=ACTION_MOVE, head_hidden_size=32)


@pytest.fixture
def model(config: ModelConfig) -> ActorCritic:
    """Create a test model."""
    return ActorCritic(config)


class TestCommandMasking:
    """Tests for action mask on command head."""

    def test_masked_action_not_selected(self, model: ActorCritic, config: ModelConfig):
        """Verify that masked actions are never sampled."""
        batch_size = 100
        obs = torch.randn(batch_size, config.obs_size)

        # Mask out ATTACK_Z1 and ATTACK_Z2, only MOVE valid
        action_mask = torch.zeros(batch_size, config.num_commands, dtype=torch.bool)
        action_mask[:, ACTION_MOVE] = True

        # Sample many times
        for _ in range(10):
            output = model(obs, action_mask=action_mask)
            # All commands should be MOVE
            assert (output.command.action == ACTION_MOVE).all(), "Masked actions were selected!"

    def test_masked_action_logits_are_negative_inf(self, model: ActorCritic, config: ModelConfig):
        """Verify that masked action logits are set to -inf."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Mask out ATTACK_Z2
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        action_mask[:, ACTION_ATTACK_Z2] = False

        output = model.command_head(obs=obs, mask=action_mask)
        logits = output.distribution.logits

        # ATTACK_Z2 logits should be -inf
        assert (logits[:, ACTION_ATTACK_Z2] == float("-inf")).all()
        # Other logits should be finite
        assert torch.isfinite(logits[:, ACTION_MOVE]).all()
        assert torch.isfinite(logits[:, ACTION_ATTACK_Z1]).all()

    def test_masked_action_zero_probability(self, model: ActorCritic, config: ModelConfig):
        """Verify that masked actions have zero probability."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Mask out ATTACK_Z1
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        action_mask[:, ACTION_ATTACK_Z1] = False

        output = model.command_head(obs=obs, mask=action_mask)
        probs = output.distribution.probs

        # ATTACK_Z1 should have zero probability
        assert (probs[:, ACTION_ATTACK_Z1] == 0.0).all()
        # Other actions should have non-zero probability
        assert (probs[:, ACTION_MOVE] > 0.0).all()
        assert (probs[:, ACTION_ATTACK_Z2] > 0.0).all()

    def test_all_actions_valid(self, model: ActorCritic, config: ModelConfig):
        """Verify all actions can be sampled when none are masked."""
        batch_size = 1000
        obs = torch.randn(batch_size, config.obs_size)
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)
        unique_commands = output.command.action.unique()

        # With enough samples, all actions should appear
        assert len(unique_commands) == config.num_commands, "Not all valid actions were sampled"


class TestAngleLossMasking:
    """Tests for angle loss masking when command != MOVE."""

    def test_angle_loss_zero_when_no_moves(self, model: ActorCritic, config: ModelConfig):
        """Verify angle loss is zero when no MOVE commands."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)

        # All ATTACK commands (no MOVE)
        commands = torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()
        assert move_mask.sum() == 0, "Test setup error: should have no MOVE commands"

        # Compute angle loss
        old_angle_log_prob = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        loss = model.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        # Loss should be zero (no MOVE commands to train on)
        assert loss.loss.item() == 0.0, f"Angle loss should be 0 when no MOVE, got {loss.loss.item()}"

    def test_angle_loss_nonzero_when_moves_present(self, model: ActorCritic, config: ModelConfig):
        """Verify angle loss is non-zero when MOVE commands present."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)

        # Half MOVE, half ATTACK
        commands = torch.tensor([ACTION_MOVE] * 4 + [ACTION_ATTACK_Z1] * 4)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()
        assert move_mask.sum() == 4, "Test setup error: should have 4 MOVE commands"

        # Compute angle loss with non-trivial advantages
        old_angle_log_prob = output.angle.log_prob.detach() + 0.1  # Slightly different
        advantages = torch.ones(batch_size)  # Positive advantages

        loss = model.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        # Loss should be non-zero
        assert loss.loss.item() != 0.0, "Angle loss should be non-zero when MOVE present"

    def test_angle_loss_only_from_move_commands(self, model: ActorCritic, config: ModelConfig):
        """Verify angle loss only comes from MOVE commands, not ATTACK."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)

        # First 4 MOVE, last 4 ATTACK
        commands = torch.tensor([ACTION_MOVE] * 4 + [ACTION_ATTACK_Z1] * 4)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()

        # Create advantages where ATTACK has huge values that would dominate if not masked
        advantages = torch.zeros(batch_size)
        advantages[:4] = 1.0  # Small advantages for MOVE
        advantages[4:] = 1000.0  # Huge advantages for ATTACK (should be ignored)

        old_angle_log_prob = output.angle.log_prob.detach()

        loss = model.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        # If ATTACK advantages were included, loss would be huge
        # Since they're masked, loss should be small (around 0 for same log_probs)
        assert abs(loss.loss.item()) < 10.0, f"ATTACK advantages leaked into loss: {loss.loss.item()}"


class TestEntropyMasking:
    """Tests for entropy masking in total_entropy."""

    def test_total_entropy_masked_excludes_angle_for_attack(self, model: ActorCritic, config: ModelConfig):
        """Verify total_entropy excludes angle entropy for ATTACK commands."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # All ATTACK commands
        commands = torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()
        assert move_mask.sum() == 0

        # Masked entropy should equal command entropy only
        masked_entropy = output.total_entropy(move_mask)
        expected_entropy = output.command.entropy

        assert torch.allclose(masked_entropy, expected_entropy), (
            f"Masked entropy {masked_entropy} != command entropy {expected_entropy}"
        )

    def test_total_entropy_masked_includes_angle_for_move(self, model: ActorCritic, config: ModelConfig):
        """Verify total_entropy includes angle entropy for MOVE commands."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # All MOVE commands
        commands = torch.full((batch_size,), ACTION_MOVE, dtype=torch.long)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()
        assert move_mask.sum() == batch_size

        # Masked entropy should equal command + angle entropy
        masked_entropy = output.total_entropy(move_mask)
        expected_entropy = output.command.entropy + output.angle.entropy

        assert torch.allclose(masked_entropy, expected_entropy), (
            f"Masked entropy {masked_entropy} != cmd+angle entropy {expected_entropy}"
        )

    def test_total_entropy_partial_masking(self, model: ActorCritic, config: ModelConfig):
        """Verify total_entropy correctly handles mixed commands."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Mixed: [MOVE, ATTACK, MOVE, ATTACK]
        commands = torch.tensor([ACTION_MOVE, ACTION_ATTACK_Z1, ACTION_MOVE, ACTION_ATTACK_Z2])

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()

        masked_entropy = output.total_entropy(move_mask)

        # Manually compute expected entropy
        expected = output.command.entropy + output.angle.entropy * move_mask

        assert torch.allclose(masked_entropy, expected), f"Masked entropy {masked_entropy} != expected {expected}"

    def test_total_entropy_requires_mask(self, model: ActorCritic, config: ModelConfig):
        """Verify total_entropy fails fast when mask is not provided."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        # Calling without mask should raise TypeError (missing required argument)
        with pytest.raises(TypeError):
            output.total_entropy()

    def test_total_log_prob_requires_mask(self, model: ActorCritic, config: ModelConfig):
        """Verify total_log_prob fails fast when mask is not provided."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        # Calling without mask should raise TypeError (missing required argument)
        with pytest.raises(TypeError):
            output.total_log_prob()


class TestLogProbMasking:
    """Tests for log probability masking in total_log_prob."""

    def test_total_log_prob_masked_excludes_angle_for_attack(self, model: ActorCritic, config: ModelConfig):
        """Verify total_log_prob excludes angle log_prob for ATTACK commands."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # All ATTACK commands
        commands = torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()

        # Masked log_prob should equal command log_prob only
        masked_log_prob = output.total_log_prob(move_mask)
        expected_log_prob = output.command.log_prob

        assert torch.allclose(masked_log_prob, expected_log_prob), (
            f"Masked log_prob {masked_log_prob} != command log_prob {expected_log_prob}"
        )

    def test_total_log_prob_masked_includes_angle_for_move(self, model: ActorCritic, config: ModelConfig):
        """Verify total_log_prob includes angle log_prob for MOVE commands."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # All MOVE commands
        commands = torch.full((batch_size,), ACTION_MOVE, dtype=torch.long)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()

        # Masked log_prob should equal command + angle log_prob
        masked_log_prob = output.total_log_prob(move_mask)
        expected_log_prob = output.command.log_prob + output.angle.log_prob

        assert torch.allclose(masked_log_prob, expected_log_prob), (
            f"Masked log_prob {masked_log_prob} != cmd+angle log_prob {expected_log_prob}"
        )

    def test_importance_ratio_correct_for_attack(self, model: ActorCritic, config: ModelConfig):
        """Verify importance ratio only uses command for ATTACK transitions."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        commands = torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()

        # Simulate old log probs (behavior policy)
        old_cmd_log_prob = output.command.log_prob.detach() - 0.1
        old_angle_log_prob = torch.randn(batch_size)  # Random, shouldn't matter

        # Compute importance ratio with masking
        new_total = output.total_log_prob(move_mask)
        old_total = old_cmd_log_prob + old_angle_log_prob * move_mask

        ratio = torch.exp(new_total - old_total)

        # Expected ratio: only command ratio matters
        expected_ratio = torch.exp(output.command.log_prob - old_cmd_log_prob)

        assert torch.allclose(ratio, expected_ratio, atol=1e-5), (
            f"Importance ratio {ratio} != expected {expected_ratio}"
        )


class TestGradientFlow:
    """Tests for correct gradient flow with masking."""

    def test_no_gradient_to_angle_head_when_no_moves(self, model: ActorCritic, config: ModelConfig):
        """Verify no gradients flow to angle head when all commands are ATTACK."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)

        # All ATTACK commands
        commands = torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()

        # Compute angle loss (should be zero)
        old_angle_log_prob = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        angle_loss = model.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        # Also use masked entropy
        entropy = output.total_entropy(move_mask).mean()

        # Backward through angle loss and entropy
        total_loss = angle_loss.loss - 0.01 * entropy
        model.zero_grad()
        total_loss.backward()

        # Check angle head gradients are zero (or None)
        for name, param in model.angle_head.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm == 0.0, f"Angle head {name} has non-zero gradient: {grad_norm}"

    def test_gradient_flows_to_angle_head_when_moves_present(self, model: ActorCritic, config: ModelConfig):
        """Verify gradients flow to angle head when MOVE commands present."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)

        # Half MOVE, half ATTACK
        commands = torch.tensor([ACTION_MOVE] * 4 + [ACTION_ATTACK_Z1] * 4)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()

        # Compute angle loss
        old_angle_log_prob = output.angle.log_prob.detach() + 0.1
        advantages = torch.ones(batch_size)

        angle_loss = model.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        # Backward
        model.zero_grad()
        angle_loss.loss.backward()

        # Check angle head has non-zero gradients
        has_gradient = False
        for _, param in model.angle_head.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 0:
                has_gradient = True
                break

        assert has_gradient, "Angle head should have gradients when MOVE commands present"

    def test_command_head_always_has_gradients(self, model: ActorCritic, config: ModelConfig):
        """Verify command head always receives gradients regardless of command type."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)

        # All ATTACK (angle masked)
        commands = torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, action_mask=action_mask)

        # Compute command loss
        old_cmd_log_prob = output.command.log_prob.detach() + 0.1
        advantages = torch.ones(batch_size)

        move_mask = torch.ones(batch_size)  # All samples used for command head
        cmd_loss = model.command_head.compute_loss(
            new_log_prob=output.command.log_prob,
            old_log_prob=old_cmd_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        model.zero_grad()
        cmd_loss.loss.backward()

        # Check command head has non-zero gradients
        has_gradient = False
        for _, param in model.command_head.named_parameters():
            if param.grad is not None and param.grad.norm().item() > 0:
                has_gradient = True
                break

        assert has_gradient, "Command head should always have gradients"


class TestAngleHeadIndependence:
    """Tests verifying angle head is independent of command (AlphaZero-style)."""

    def test_angle_output_same_regardless_of_command(self, model: ActorCritic, config: ModelConfig):
        """Verify angle output is the same regardless of which command is selected."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Same angle for evaluation
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        # Forward with MOVE command
        output_move = model(
            obs,
            command=torch.full((batch_size,), ACTION_MOVE, dtype=torch.long),
            angle=angles,
            action_mask=action_mask,
        )

        # Forward with ATTACK command
        output_attack = model(
            obs,
            command=torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long),
            angle=angles,
            action_mask=action_mask,
        )

        # Angle outputs should be identical (same obs, same angle to evaluate)
        assert torch.allclose(output_move.angle.log_prob, output_attack.angle.log_prob), (
            "Angle log_prob should be independent of command"
        )
        assert torch.allclose(output_move.angle.entropy, output_attack.angle.entropy), (
            "Angle entropy should be independent of command"
        )

    def test_angle_head_input_size_is_obs_only(self, config: ModelConfig):
        """Verify angle head input size equals obs_size (no command conditioning)."""
        angle_head = AngleHead(config)

        # Check the first linear layer input size (encoder is the first part of the network)
        first_layer = angle_head.encoder[0]
        assert first_layer.in_features == config.obs_size, (
            f"Angle head input size {first_layer.in_features} != obs_size {config.obs_size}"
        )

    def test_deterministic_angle_independent_of_command(self, model: ActorCritic, config: ModelConfig):
        """Verify deterministic angle is the same regardless of command."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Get deterministic angle (should only depend on obs)
        angle = model.angle_head.get_deterministic_action(obs)

        # Call again - should be identical
        angle2 = model.angle_head.get_deterministic_action(obs)

        assert torch.allclose(angle, angle2), "Deterministic angle should be consistent"


class TestEndToEndTraining:
    """Integration tests for full training step with masking."""

    def test_full_training_step_mixed_commands(self, model: ActorCritic, config: ModelConfig):
        """Test a complete training step with mixed commands."""
        batch_size = 16
        obs = torch.randn(batch_size, config.obs_size)

        # Mixed commands
        commands = torch.tensor([ACTION_MOVE, ACTION_ATTACK_Z1, ACTION_ATTACK_Z2, ACTION_MOVE] * 4)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        # Forward pass
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        # Compute move mask
        move_mask = (commands == ACTION_MOVE).float()

        # Simulate old log probs
        old_cmd_log_prob = output.command.log_prob.detach() + torch.randn(batch_size) * 0.1
        old_angle_log_prob = output.angle.log_prob.detach() + torch.randn(batch_size) * 0.1

        # Advantages and targets
        advantages = torch.randn(batch_size)
        vtrace_targets = torch.randn(batch_size)

        # Compute all losses
        losses = model.compute_losses(
            output=output,
            old_cmd_log_prob=old_cmd_log_prob,
            old_angle_log_prob=old_angle_log_prob,
            advantages=advantages,
            vtrace_targets=vtrace_targets,
            move_mask=move_mask,
            clip_epsilon=0.2,
        )

        # Compute total loss with masked entropy
        entropy = output.total_entropy(move_mask).mean()
        total_loss = losses["command"].loss + losses["angle"].loss + 0.5 * losses["value"].loss - 0.01 * entropy

        # Backward pass
        model.zero_grad()
        total_loss.backward()

        # Verify all parameters have gradients (except aux_head which requires aux_loss)
        for name, param in model.named_parameters():
            if "aux_head" in name:
                # aux_head only gets gradients when aux_loss is computed
                continue
            assert param.grad is not None, f"Parameter {name} has no gradient"

        # Verify loss is finite
        assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss}"

    def test_training_step_all_moves(self, model: ActorCritic, config: ModelConfig):
        """Test training step when all commands are MOVE."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)
        commands = torch.full((batch_size,), ACTION_MOVE, dtype=torch.long)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()
        assert move_mask.sum() == batch_size

        # Entropy should include both command and angle
        entropy = output.total_entropy(move_mask)
        expected = output.command.entropy + output.angle.entropy
        assert torch.allclose(entropy, expected)

        # Angle loss should be non-zero
        old_angle_log_prob = output.angle.log_prob.detach() + 0.1
        advantages = torch.ones(batch_size)

        angle_loss = model.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        assert angle_loss.loss.item() != 0.0

    def test_training_step_all_attacks(self, model: ActorCritic, config: ModelConfig):
        """Test training step when all commands are ATTACK."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)
        commands = torch.full((batch_size,), ACTION_ATTACK_Z1, dtype=torch.long)
        angles = torch.randn(batch_size, 2)
        angles = angles / angles.norm(dim=-1, keepdim=True)

        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)
        output = model(obs, command=commands, angle=angles, action_mask=action_mask)

        move_mask = (commands == ACTION_MOVE).float()
        assert move_mask.sum() == 0

        # Entropy should only be command entropy
        entropy = output.total_entropy(move_mask)
        expected = output.command.entropy  # angle entropy * 0 = 0
        assert torch.allclose(entropy, expected)

        # Angle loss should be zero
        old_angle_log_prob = output.angle.log_prob.detach() + 0.1
        advantages = torch.ones(batch_size) * 100  # Large advantages

        angle_loss = model.angle_head.compute_loss(
            new_log_prob=output.angle.log_prob,
            old_log_prob=old_angle_log_prob,
            advantages=advantages,
            clip_epsilon=0.2,
            mask=move_mask,
        )

        assert angle_loss.loss.item() == 0.0, f"Angle loss should be 0 for all ATTACKs, got {angle_loss.loss.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
