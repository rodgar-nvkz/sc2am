"""Tests for auxiliary prediction task and von Mises angle distribution.

Verifies that:
1. Auxiliary head correctly predicts observation features
2. Auxiliary loss gradient flows through encoder
3. Encoder representations change based on aux loss
4. Disabling aux_enabled works correctly
5. Von Mises distribution provides proper angular exploration
6. Action output format is correct (sin, cos)
"""

import math

import torch
import torch.nn.functional as F

from smarte.smarte_v3.model import ActorCritic, ModelConfig


class TestAuxiliaryPrediction:
    """Tests for auxiliary prediction task."""

    def test_aux_loss_computes(self):
        """Auxiliary loss should compute without errors."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
            aux_target_indices=[7, 8, 9, 14, 15, 16],
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        aux_loss = model.compute_aux_loss(obs)

        assert aux_loss.shape == (), "Aux loss should be scalar"
        assert aux_loss.item() >= 0, "MSE loss should be non-negative"
        assert not torch.isnan(aux_loss), "Aux loss should not be NaN"

    def test_aux_loss_decreases_with_training(self):
        """Auxiliary loss should decrease when trained."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
            aux_target_indices=[7, 8, 9, 14, 15, 16],
        )
        model = ActorCritic(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Fixed batch of observations
        obs = torch.randn(32, 20)

        initial_loss = model.compute_aux_loss(obs).item()

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            loss = model.compute_aux_loss(obs)
            loss.backward()
            optimizer.step()

        final_loss = model.compute_aux_loss(obs).item()

        assert final_loss < initial_loss * 0.5, (
            f"Aux loss should decrease significantly: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_aux_gradient_flows_to_encoder(self):
        """Auxiliary loss gradients should flow through encoder."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
            aux_target_indices=[7, 8, 9],
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        aux_loss = model.compute_aux_loss(obs)
        aux_loss.backward()

        # Check encoder gradients exist
        encoder = model.angle_head.encoder
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Encoder param {name} should have gradient"
            assert param.grad.abs().sum() > 0, f"Encoder param {name} gradient should be non-zero"

    def test_aux_disabled_returns_zero(self):
        """When aux_enabled=False, aux loss should be zero."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=False,
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        aux_loss = model.compute_aux_loss(obs)

        assert aux_loss.item() == 0.0, "Aux loss should be zero when disabled"

    def test_aux_head_not_created_when_disabled(self):
        """When aux_enabled=False, aux_head should not exist."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=False,
        )
        model = ActorCritic(config)

        assert not hasattr(model.angle_head, "aux_head") or not model.angle_head.aux_enabled

    def test_encoder_representations_differ_with_aux(self):
        """With aux training, encoder should produce different representations for different obs."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
            aux_target_indices=[7, 8, 9, 14, 15, 16],
        )
        model = ActorCritic(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Create observations with different enemy angles
        obs1 = torch.zeros(1, 20)
        obs1[0, 7:10] = torch.tensor([1.0, 0.0, 0.5])  # Enemy at angle 0, distance 0.5

        obs2 = torch.zeros(1, 20)
        obs2[0, 7:10] = torch.tensor([0.0, 1.0, 0.5])  # Enemy at angle π/2, distance 0.5

        # Train on diverse observations
        diverse_obs = torch.randn(64, 20)
        for _ in range(100):
            optimizer.zero_grad()
            loss = model.compute_aux_loss(diverse_obs)
            loss.backward()
            optimizer.step()

        # Check that encoder produces different hidden states
        with torch.no_grad():
            h1 = model.angle_head.encoder(obs1)
            h2 = model.angle_head.encoder(obs2)

            # Hidden states should be different
            diff = (h1 - h2).abs().mean().item()
            assert diff > 0.1, f"Encoder should differentiate observations, got diff={diff:.4f}"

    def test_aux_predicts_correct_features(self):
        """After training, aux head should accurately predict target features."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
            aux_target_indices=[7, 8, 9],  # z1 angle_sin, angle_cos, distance
        )
        model = ActorCritic(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Train on random observations
        for _ in range(200):
            obs = torch.randn(32, 20)
            optimizer.zero_grad()
            loss = model.compute_aux_loss(obs)
            loss.backward()
            optimizer.step()

        # Test prediction accuracy on new observations
        model.eval()
        with torch.no_grad():
            test_obs = torch.randn(16, 20)
            h = model.angle_head.encoder(test_obs)
            aux_pred = model.angle_head.aux_head(h)
            aux_targets = test_obs[:, config.aux_target_indices]

            mse = F.mse_loss(aux_pred, aux_targets).item()
            assert mse < 0.1, f"Aux prediction should be accurate, got MSE={mse:.4f}"


class TestVonMisesDistribution:
    """Tests for von Mises angle distribution."""

    def test_action_output_shape(self):
        """Action output should be (B, 2) for sin/cos."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        action_mask = torch.ones(8, 3, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        assert output.angle.action.shape == (8, 2), f"Expected (8, 2), got {output.angle.action.shape}"

    def test_action_is_unit_vector(self):
        """Action (sin, cos) should form approximately unit vectors."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)

        obs = torch.randn(16, 20)
        action_mask = torch.ones(16, 3, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)
        action = output.angle.action

        # Compute magnitude of each (sin, cos) pair
        magnitudes = torch.sqrt(action[:, 0] ** 2 + action[:, 1] ** 2)

        # Should be close to 1.0 (unit circle)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), (
            f"Actions should be unit vectors, got magnitudes: {magnitudes}"
        )

    def test_log_prob_shape(self):
        """Log prob should be scalar per batch element."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        action_mask = torch.ones(8, 3, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        assert output.angle.log_prob.shape == (8,), f"Expected (8,), got {output.angle.log_prob.shape}"

    def test_entropy_shape(self):
        """Entropy should be scalar per batch element."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        action_mask = torch.ones(8, 3, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        assert output.angle.entropy.shape == (8,), f"Expected (8,), got {output.angle.entropy.shape}"

    def test_evaluate_existing_action(self):
        """Should be able to evaluate log_prob of provided actions."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        action_mask = torch.ones(8, 3, dtype=torch.bool)

        # Create some actions (sin, cos pairs)
        angles = torch.randn(8) * math.pi  # Random angles
        actions = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)

        output = model(obs, angle=actions, action_mask=action_mask)

        assert output.angle.log_prob.shape == (8,)
        assert not torch.isnan(output.angle.log_prob).any(), "Log prob should not contain NaN"

    def test_deterministic_action(self):
        """Deterministic action should be reproducible."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)
        model.eval()

        obs = torch.randn(4, 20)
        action_mask = torch.ones(4, 3, dtype=torch.bool)

        command, angle = model.get_deterministic_action(obs, action_mask=action_mask)

        assert angle.shape == (4, 2)

        # Should be reproducible
        command2, angle2 = model.get_deterministic_action(obs, action_mask=action_mask)
        assert torch.allclose(angle, angle2), "Deterministic action should be reproducible"

    def test_deterministic_action_is_unit_vector(self):
        """Deterministic action should also be unit vector."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)
        model.eval()

        obs = torch.randn(8, 20)
        action_mask = torch.ones(8, 3, dtype=torch.bool)

        _, angle = model.get_deterministic_action(obs, action_mask=action_mask)

        magnitudes = torch.sqrt(angle[:, 0] ** 2 + angle[:, 1] ** 2)
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)

    def test_concentration_is_positive(self):
        """Concentration parameter should always be positive."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            angle_init_log_concentration=-5.0,  # Very low initial value
        )
        model = ActorCritic(config)

        concentration = model.angle_head.get_concentration()
        assert concentration > 0, f"Concentration should be positive, got {concentration}"

    def test_angular_exploration_coverage(self):
        """Von Mises should explore full angle range, unlike Gaussian on (sin, cos)."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            angle_init_log_concentration=0.0,  # κ ≈ 1, moderate exploration
        )
        model = ActorCritic(config)

        obs = torch.randn(1, 20).expand(1000, -1)  # Same obs, many samples
        action_mask = torch.ones(1000, 3, dtype=torch.bool)

        # Sample many actions
        with torch.no_grad():
            output = model(obs, action_mask=action_mask)
            actions = output.angle.action

        # Convert to angles
        angles = torch.atan2(actions[:, 0], actions[:, 1])

        # Check that we explore a wide range of angles
        angle_range = angles.max() - angles.min()

        # With κ=1, we should explore most of the circle
        # Range should be at least 4 radians (about 230 degrees)
        assert angle_range > 4.0, f"Expected wide angular exploration, got range={angle_range:.2f} radians"

    def test_policy_gradient_flows(self):
        """Gradients should flow through the policy for PPO updates."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        action_mask = torch.ones(8, 3, dtype=torch.bool)

        # Forward pass
        output = model(obs, action_mask=action_mask)

        # Simulate policy gradient loss
        advantages = torch.randn(8)
        loss = -(output.angle.log_prob * advantages).mean()

        loss.backward()

        # Check gradients exist for output head
        for name, param in model.angle_head.output_head.named_parameters():
            assert param.grad is not None, f"Output head param {name} should have gradient"

        # Check gradient for concentration parameter
        assert model.angle_head.log_concentration.grad is not None, "Concentration parameter should have gradient"


class TestIntegration:
    """Integration tests for full model with von Mises."""

    def test_full_forward_pass(self):
        """Full forward pass should work without errors."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
            aux_target_indices=[7, 8, 9, 14, 15, 16],
        )
        model = ActorCritic(config)

        obs = torch.randn(8, 20)
        action_mask = torch.ones(8, 3, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        # Check all outputs
        assert output.command.action.shape == (8,)
        assert output.command.log_prob.shape == (8,)
        assert output.angle.action.shape == (8, 2)
        assert output.angle.log_prob.shape == (8,)
        assert output.value.shape == (8,)

    def test_training_step(self):
        """Complete training step should work."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
        )
        model = ActorCritic(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        obs = torch.randn(16, 20)
        action_mask = torch.ones(16, 3, dtype=torch.bool)

        # Forward pass
        output = model(obs, action_mask=action_mask)

        # Compute losses
        advantages = torch.randn(16)
        move_mask = (output.command.action == 0).float()

        policy_loss = -(output.angle.log_prob * advantages * move_mask).sum() / move_mask.sum().clamp(min=1)
        entropy_loss = -output.angle.entropy.mean()
        aux_loss = model.compute_aux_loss(obs)

        total_loss = policy_loss + 0.01 * entropy_loss + 0.5 * aux_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Should complete without error
        assert True

    def test_model_save_load(self):
        """Model should be saveable and loadable."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
        )
        model = ActorCritic(config)

        # Get initial output
        obs = torch.randn(4, 20)
        action_mask = torch.ones(4, 3, dtype=torch.bool)

        with torch.no_grad():
            _, initial_angle = model.get_deterministic_action(obs, action_mask=action_mask)

        # Save and load state dict
        state_dict = model.state_dict()

        model2 = ActorCritic(config)
        model2.load_state_dict(state_dict)

        with torch.no_grad():
            _, loaded_angle = model2.get_deterministic_action(obs, action_mask=action_mask)

        assert torch.allclose(initial_angle, loaded_angle), "Loaded model should produce same output"
