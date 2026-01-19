"""Tests for auxiliary prediction task in AngleHead.

Verifies that:
1. Auxiliary head correctly predicts observation features
2. Auxiliary loss gradient flows through encoder
3. Encoder representations change based on aux loss
4. Disabling aux_enabled works correctly
"""

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

    def test_policy_still_works_with_aux(self):
        """Policy should still produce valid actions with aux task enabled."""
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

        # Check command output
        assert output.command.action.shape == (8,)
        assert output.command.log_prob.shape == (8,)
        assert output.command.entropy.shape == (8,)

        # Check angle output
        assert output.angle.action.shape == (8, 2)
        assert output.angle.log_prob.shape == (8,)
        assert output.angle.entropy.shape == (8,)

        # Check value output (CriticHead returns (B,) not (B, 1))
        assert output.value.shape == (8,)

    def test_deterministic_action_works_with_aux(self):
        """Deterministic action selection should work with aux task enabled."""
        config = ModelConfig(
            obs_size=20,
            num_commands=3,
            move_action_id=0,
            aux_enabled=True,
            aux_target_indices=[7, 8, 9],
        )
        model = ActorCritic(config)
        model.eval()

        obs = torch.randn(4, 20)
        action_mask = torch.ones(4, 3, dtype=torch.bool)

        command, angle = model.get_deterministic_action(obs, action_mask=action_mask)

        assert command.shape == (4,)
        assert angle.shape == (4, 2)

        # Deterministic should be reproducible
        command2, angle2 = model.get_deterministic_action(obs, action_mask=action_mask)
        assert torch.equal(command, command2)
        assert torch.allclose(angle, angle2)
