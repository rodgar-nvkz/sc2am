"""Tests verifying model outputs align with environment expectations.

These tests ensure that:
1. Model action space matches environment action space
2. Model output shapes and types are compatible with env.step()
3. Action constants are consistent between model and environment
4. Action masking from environment is compatible with model
5. Angle representation (sin, cos) is handled consistently
"""

import numpy as np
import pytest
import torch

from smarte.smarte_v3.env import SC2GymEnv
from smarte.smarte_v3.model.actor_critic import ActorCritic
from smarte.smarte_v3.model.config import ModelConfig

# Import constants from environment class
ACTION_MOVE = SC2GymEnv.ACTION_MOVE
ACTION_ATTACK_Z1 = SC2GymEnv.ACTION_ATTACK_Z1
ACTION_ATTACK_Z2 = SC2GymEnv.ACTION_ATTACK_Z2
NUM_COMMANDS = SC2GymEnv.NUM_COMMANDS
OBS_SIZE = SC2GymEnv.OBS_SIZE


@pytest.fixture
def config() -> ModelConfig:
    """Create config matching environment specs."""
    return ModelConfig(obs_size=OBS_SIZE, num_commands=NUM_COMMANDS, move_action_id=ACTION_MOVE)


@pytest.fixture
def model(config: ModelConfig) -> ActorCritic:
    """Create model with environment-compatible config."""
    return ActorCritic(config)


class TestActionSpaceAlignment:
    """Tests for action space compatibility between model and environment."""

    def test_num_commands_matches(self, config: ModelConfig):
        """Verify model num_commands matches environment NUM_COMMANDS."""
        assert config.num_commands == NUM_COMMANDS, (
            f"Model num_commands ({config.num_commands}) != env NUM_COMMANDS ({NUM_COMMANDS})"
        )

    def test_obs_size_matches(self, config: ModelConfig):
        """Verify model obs_size matches environment OBS_SIZE."""
        assert config.obs_size == OBS_SIZE, f"Model obs_size ({config.obs_size}) != env OBS_SIZE ({OBS_SIZE})"

    def test_action_constants_defined(self):
        """Verify all expected action constants are defined in environment."""
        assert ACTION_MOVE == 0, "ACTION_MOVE should be 0"
        assert ACTION_ATTACK_Z1 == 1, "ACTION_ATTACK_Z1 should be 1"
        assert ACTION_ATTACK_Z2 == 2, "ACTION_ATTACK_Z2 should be 2"
        assert NUM_COMMANDS == 3, "NUM_COMMANDS should be 3"

    def test_command_output_is_valid_index(self, model: ActorCritic, config: ModelConfig):
        """Verify model command output is a valid action index."""
        batch_size = 10
        obs = torch.randn(batch_size, config.obs_size)
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        commands = output.command.action
        assert commands.dtype == torch.int64, f"Commands should be int64, got {commands.dtype}"
        assert (commands >= 0).all(), "Commands should be >= 0"
        assert (commands < NUM_COMMANDS).all(), f"Commands should be < {NUM_COMMANDS}"

    def test_angle_output_shape(self, model: ActorCritic, config: ModelConfig):
        """Verify angle output has correct shape (B, 2) for (sin, cos)."""
        batch_size = 5
        obs = torch.randn(batch_size, config.obs_size)
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        angles = output.angle.action
        assert angles.shape == (batch_size, 2), f"Angle shape should be ({batch_size}, 2), got {angles.shape}"

    def test_angle_is_unit_vector(self, model: ActorCritic, config: ModelConfig):
        """Verify angle output is normalized to unit circle."""
        batch_size = 100
        obs = torch.randn(batch_size, config.obs_size)
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        angles = output.angle.action
        norms = torch.norm(angles, dim=-1)

        # Should be close to 1.0 (unit vectors)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"Angle vectors should be unit length, got norms: {norms}"
        )

    def test_angle_values_in_valid_range(self, model: ActorCritic, config: ModelConfig):
        """Verify angle (sin, cos) values are in [-1, 1]."""
        batch_size = 100
        obs = torch.randn(batch_size, config.obs_size)
        action_mask = torch.ones(batch_size, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        angles = output.angle.action
        assert (angles >= -1.0 - 1e-5).all(), "Angle values should be >= -1"
        assert (angles <= 1.0 + 1e-5).all(), "Angle values should be <= 1"


class TestActionMaskAlignment:
    """Tests for action mask compatibility."""

    def test_action_mask_shape(self, model: ActorCritic, config: ModelConfig):
        """Verify model accepts action mask with correct shape."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Environment returns mask of shape (NUM_COMMANDS,) per step
        # For batch, it's (batch_size, NUM_COMMANDS)
        action_mask = torch.ones(batch_size, NUM_COMMANDS, dtype=torch.bool)

        # Should not raise
        output = model(obs, action_mask=action_mask)
        assert output is not None

    def test_action_mask_dtype_bool(self, model: ActorCritic, config: ModelConfig):
        """Verify model accepts boolean action mask (environment format)."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Environment returns np.ndarray with dtype=bool, converted to torch bool
        action_mask = torch.ones(batch_size, NUM_COMMANDS, dtype=torch.bool)
        action_mask[:, ACTION_ATTACK_Z2] = False  # Z2 out of range

        output = model(obs, action_mask=action_mask)

        # Should never select masked action
        assert (output.command.action != ACTION_ATTACK_Z2).all()

    def test_all_attacks_masked_forces_move(self, model: ActorCritic, config: ModelConfig):
        """When both attacks are masked, model should only output MOVE."""
        batch_size = 50
        obs = torch.randn(batch_size, config.obs_size)

        action_mask = torch.zeros(batch_size, NUM_COMMANDS, dtype=torch.bool)
        action_mask[:, ACTION_MOVE] = True  # Only MOVE valid

        output = model(obs, action_mask=action_mask)

        assert (output.command.action == ACTION_MOVE).all(), "When only MOVE is valid, all commands should be MOVE"


class TestEnvironmentActionFormat:
    """Tests for action format expected by environment."""

    def test_action_dict_format(self, model: ActorCritic, config: ModelConfig):
        """Verify model output can be converted to env action dict format."""
        obs = torch.randn(1, config.obs_size)
        action_mask = torch.ones(1, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        # This is the format expected by env.step()
        action = {
            "command": output.command.action.item(),
            "angle": output.angle.action.squeeze(0).numpy(),
        }

        # Verify types
        assert isinstance(action["command"], int), "command should be int"
        assert isinstance(action["angle"], np.ndarray), "angle should be numpy array"
        assert action["angle"].shape == (2,), "angle should have shape (2,)"
        assert action["angle"].dtype == np.float32, f"angle dtype should be float32, got {action['angle'].dtype}"

    def test_command_item_extracts_scalar(self, model: ActorCritic, config: ModelConfig):
        """Verify command.action.item() gives Python int."""
        obs = torch.randn(1, config.obs_size)
        action_mask = torch.ones(1, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        command = output.command.action.item()
        assert isinstance(command, int), f"command.item() should be int, got {type(command)}"
        assert 0 <= command < NUM_COMMANDS

    def test_angle_squeeze_gives_1d(self, model: ActorCritic, config: ModelConfig):
        """Verify angle.squeeze(0).numpy() gives 1D array for single obs."""
        obs = torch.randn(1, config.obs_size)
        action_mask = torch.ones(1, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        angle = output.angle.action.squeeze(0).numpy()
        assert angle.ndim == 1, f"angle should be 1D after squeeze, got {angle.ndim}D"
        assert angle.shape == (2,), f"angle shape should be (2,), got {angle.shape}"


class TestAngleRepresentationAlignment:
    """Tests for angle (sin, cos) representation consistency."""

    def test_angle_order_is_sin_cos(self, model: ActorCritic, config: ModelConfig):
        """Document that angle is (sin, cos) = (dy, dx) order.

        Environment uses: dy, dx = angle_sincos
        This means angle[0] = sin (y component), angle[1] = cos (x component)
        """
        # This is a documentation test - the actual order is defined by convention
        # The environment code shows: dy, dx = angle_sincos
        # So angle[0] = dy = sin, angle[1] = dx = cos

        obs = torch.randn(1, config.obs_size)
        action_mask = torch.ones(1, config.num_commands, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)
        angle = output.angle.action.squeeze(0)

        sin_component = angle[0]
        cos_component = angle[1]

        # Verify it's a valid unit vector
        magnitude = torch.sqrt(sin_component**2 + cos_component**2)
        assert torch.isclose(magnitude, torch.tensor(1.0), atol=1e-5)

    def test_deterministic_angle_is_normalized(self, model: ActorCritic, config: ModelConfig):
        """Verify deterministic angle is also normalized."""
        obs = torch.randn(10, config.obs_size)
        action_mask = torch.ones(10, config.num_commands, dtype=torch.bool)

        command, angle = model.get_deterministic_action(obs, action_mask=action_mask)

        norms = torch.norm(angle, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestObservationAlignment:
    """Tests for observation compatibility."""

    def test_model_accepts_env_obs_shape(self, model: ActorCritic, config: ModelConfig):
        """Verify model accepts observation with environment shape."""
        # Environment returns obs of shape (OBS_SIZE,)
        # For model, we unsqueeze to (1, OBS_SIZE)
        env_obs = np.random.randn(OBS_SIZE).astype(np.float32)
        obs_tensor = torch.from_numpy(env_obs).unsqueeze(0)

        action_mask = torch.ones(1, NUM_COMMANDS, dtype=torch.bool)

        # Should not raise
        output = model(obs_tensor, action_mask=action_mask)
        assert output is not None

    def test_model_handles_batched_obs(self, model: ActorCritic, config: ModelConfig):
        """Verify model handles batched observations for training."""
        batch_size = 32
        obs = torch.randn(batch_size, OBS_SIZE)
        action_mask = torch.ones(batch_size, NUM_COMMANDS, dtype=torch.bool)

        output = model(obs, action_mask=action_mask)

        assert output.command.action.shape == (batch_size,)
        assert output.angle.action.shape == (batch_size, 2)
        assert output.value.shape == (batch_size,)


class TestCollectorIntegration:
    """Tests mimicking the collector's model-environment interaction."""

    def test_collector_action_flow(self, model: ActorCritic, config: ModelConfig):
        """Simulate the exact flow used in collector.py."""
        # Simulate environment observation
        obs = np.random.randn(OBS_SIZE).astype(np.float32)
        action_mask = np.ones(NUM_COMMANDS, dtype=bool)
        action_mask[ACTION_ATTACK_Z2] = False  # Z2 out of range

        # Convert to tensors (as done in collector)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        mask_tensor = torch.from_numpy(action_mask).unsqueeze(0)

        # Get model output
        with torch.no_grad():
            output = model(obs_tensor, action_mask=mask_tensor)

        # Extract action (as done in collector)
        command = output.command.action.item()
        angle = output.angle.action.squeeze(0).numpy()

        # Build action dict (as done in collector)
        action = {"command": command, "angle": angle}

        # Verify action format matches environment expectation
        assert isinstance(action["command"], int)
        assert 0 <= action["command"] < NUM_COMMANDS
        assert action["command"] != ACTION_ATTACK_Z2  # Was masked

        assert isinstance(action["angle"], np.ndarray)
        assert action["angle"].shape == (2,)
        assert action["angle"].dtype == np.float32

    def test_episode_collection_simulation(self, model: ActorCritic, config: ModelConfig):
        """Simulate collecting an episode with various action masks."""
        num_steps = 20
        model.eval()

        # Simulate episode
        obs = np.random.randn(OBS_SIZE).astype(np.float32)

        collected_commands = []
        collected_angles = []
        collected_log_probs = []

        with torch.no_grad():
            for step in range(num_steps):
                # Simulate varying action masks
                action_mask = np.ones(NUM_COMMANDS, dtype=bool)
                if step % 3 == 0:
                    action_mask[ACTION_ATTACK_Z1] = False
                if step % 5 == 0:
                    action_mask[ACTION_ATTACK_Z2] = False

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                mask_tensor = torch.from_numpy(action_mask).unsqueeze(0)

                output = model(obs_tensor, action_mask=mask_tensor)

                command = output.command.action.item()
                angle = output.angle.action.squeeze(0).numpy()

                # Verify masked actions not selected
                if not action_mask[ACTION_ATTACK_Z1]:
                    assert command != ACTION_ATTACK_Z1
                if not action_mask[ACTION_ATTACK_Z2]:
                    assert command != ACTION_ATTACK_Z2

                collected_commands.append(command)
                collected_angles.append(angle)
                collected_log_probs.append(output.command.log_prob.item())

                # Simulate next observation
                obs = np.random.randn(OBS_SIZE).astype(np.float32)

        # Verify collected data shapes (as stored in Episode dataclass)
        commands_array = np.array(collected_commands, dtype=np.int64)
        angles_array = np.array(collected_angles, dtype=np.float32)

        assert commands_array.shape == (num_steps,)
        assert angles_array.shape == (num_steps, 2)


class TestEvalAlignment:
    """Tests for evaluation mode alignment with environment."""

    def test_deterministic_action_format(self, model: ActorCritic, config: ModelConfig):
        """Verify deterministic action has correct format for evaluation."""
        obs = torch.randn(1, config.obs_size)
        action_mask = torch.ones(1, config.num_commands, dtype=torch.bool)

        # eval.py uses torch.no_grad() context
        with torch.no_grad():
            command, angle = model.get_deterministic_action(obs, action_mask=action_mask)

            # Build action dict as done in eval.py
            action = {
                "command": command.squeeze(0).item(),
                "angle": angle.squeeze(0).numpy(),
            }

        assert isinstance(action["command"], int)
        assert isinstance(action["angle"], np.ndarray)
        assert action["angle"].shape == (2,)

    def test_deterministic_respects_mask(self, model: ActorCritic, config: ModelConfig):
        """Verify deterministic action respects action mask."""
        batch_size = 50
        obs = torch.randn(batch_size, config.obs_size)

        # Only MOVE valid
        action_mask = torch.zeros(batch_size, config.num_commands, dtype=torch.bool)
        action_mask[:, ACTION_MOVE] = True

        with torch.no_grad():
            command, angle = model.get_deterministic_action(obs, action_mask=action_mask)

        assert (command == ACTION_MOVE).all(), "Deterministic should respect mask"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
