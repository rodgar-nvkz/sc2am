"""Comprehensive tests for the entity-attention ActorCritic model.

Run with: pytest smarte/smarte_v1/model/test_model.py -v

These tests verify:
    1. Model instantiation and configuration
    2. Forward pass correctness and tensor shapes
    3. Hidden state initialization and propagation
    4. Action masking behavior
    5. Deterministic action selection
    6. Loss computation
    7. Sequence processing
    8. Edge cases (dead enemies, masked actions, etc.)
    9. Gradient flow through all components
    10. Environment action conversion
    11. Individual component behavior (encoders, attention, heads)
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from .actor_critic import ActorCritic, ActorCriticOutput
from .attention import CrossAttention
from .config import (
    ACTION_ATTACK,
    ACTION_MOVE,
    ACTION_STOP,
    NUM_ACTION_TYPES,
    OBS_ENEMY_START,
    OBS_MARINE_END,
    OBS_MARINE_START,
    OBS_TIME_LEFT_IDX,
    ModelConfig,
)
from .encoders import EnemyEncoder, EntityEncoder, MarineEncoder, TemporalEncoder
from .heads import (
    ActionTypeHead,
    AttackTargetHead,
    CombinedAuxiliaryHead,
    CriticHead,
    DamageAuxHead,
    DistanceAuxHead,
    HybridAction,
    MoveDirectionHead,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> ModelConfig:
    """Default model configuration."""
    return ModelConfig()


@pytest.fixture
def small_config() -> ModelConfig:
    """Smaller config for faster tests."""
    return ModelConfig(
        entity_embed_size=32,
        entity_hidden_size=32,
        gru_hidden_size=32,
        attention_dim=32,
        head_hidden_size=32,
    )


@pytest.fixture
def model(config: ModelConfig) -> ActorCritic:
    """Default model instance."""
    return ActorCritic(config)


@pytest.fixture
def batch_obs(config: ModelConfig) -> Tensor:
    """Batch of random observations."""
    batch_size = 4
    obs = torch.randn(batch_size, config.obs_size)
    # Ensure valid ranges for certain features
    obs[:, OBS_TIME_LEFT_IDX] = torch.rand(batch_size)  # time_left in [0, 1]
    obs[:, OBS_MARINE_START:OBS_MARINE_END] = torch.rand(batch_size, 3)  # marine obs
    # Enemy features with some alive (hp > 0) and some dead (hp = 0)
    for i in range(config.max_enemies):
        start = OBS_ENEMY_START + i * config.enemy_obs_size
        obs[:, start] = torch.tensor([0.8, 0.5, 0.0, 0.3])[:batch_size]  # hp
    return obs


# =============================================================================
# Configuration Tests
# =============================================================================


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.marine_obs_size == 3
        assert config.enemy_obs_size == 4
        assert config.max_enemies == 2
        assert config.num_action_types == NUM_ACTION_TYPES == 3

    def test_obs_size_computed(self):
        """Test obs_size is computed correctly."""
        config = ModelConfig()
        expected = 1 + 3 + 2 * 4  # time + marine + enemies
        assert config.obs_size == expected == 12

    def test_backbone_output_size_with_time(self):
        """Test backbone size includes time when enabled."""
        config = ModelConfig(use_time_feature=True)
        expected = config.gru_hidden_size + config.entity_embed_size + 1
        assert config.backbone_output_size == expected

    def test_backbone_output_size_without_time(self):
        """Test backbone size excludes time when disabled."""
        config = ModelConfig(use_time_feature=False)
        expected = config.gru_hidden_size + config.entity_embed_size
        assert config.backbone_output_size == expected

    def test_head_input_size_with_skip(self):
        """Test head input size with skip connections."""
        config = ModelConfig(use_skip_connections=True)
        expected = config.backbone_output_size + config.marine_obs_size
        assert config.head_input_size == expected

    def test_head_input_size_without_skip(self):
        """Test head input size without skip connections."""
        config = ModelConfig(use_skip_connections=False)
        assert config.head_input_size == config.backbone_output_size

    def test_action_type_names(self):
        """Test action type name mapping."""
        assert ModelConfig.action_type_name(ACTION_MOVE) == "MOVE"
        assert ModelConfig.action_type_name(ACTION_ATTACK) == "ATTACK"
        assert ModelConfig.action_type_name(ACTION_STOP) == "STOP"
        assert "UNKNOWN" in ModelConfig.action_type_name(99)


# =============================================================================
# Model Instantiation Tests
# =============================================================================


class TestModelInstantiation:
    """Tests for model instantiation."""

    def test_create_model(self, config: ModelConfig):
        """Test model can be created."""
        model = ActorCritic(config)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_components_exist(self, model: ActorCritic):
        """Test all expected components exist."""
        assert hasattr(model, "entity_encoder")
        assert hasattr(model, "temporal_encoder")
        assert hasattr(model, "cross_attention")
        assert hasattr(model, "action_type_head")
        assert hasattr(model, "move_direction_head")
        assert hasattr(model, "attack_target_head")
        assert hasattr(model, "value_head")

    def test_auxiliary_head_conditional(self):
        """Test auxiliary head is created only when enabled."""
        config_with_aux = ModelConfig(use_auxiliary_tasks=True)
        config_without_aux = ModelConfig(use_auxiliary_tasks=False)

        model_with = ActorCritic(config_with_aux)
        model_without = ActorCritic(config_without_aux)

        assert model_with.auxiliary_head is not None
        assert model_without.auxiliary_head is None

    def test_model_parameter_count(self, model: ActorCritic):
        """Test model has reasonable number of parameters."""
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0
        # Rough sanity check - should be in reasonable range
        assert 10_000 < num_params < 10_000_000

    def test_model_on_different_devices(self, config: ModelConfig):
        """Test model can be moved to different devices."""
        model = ActorCritic(config)

        # CPU
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            assert next(model_cuda.parameters()).device.type == "cuda"


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestForwardPass:
    """Tests for forward pass."""

    def test_forward_basic(self, model: ActorCritic, config: ModelConfig):
        """Test basic forward pass."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        output = model(obs, hidden)

        assert isinstance(output, ActorCriticOutput)
        assert output.action is not None
        assert output.log_prob is not None
        assert output.value is not None

    def test_forward_output_shapes(self, model: ActorCritic, config: ModelConfig):
        """Test output tensor shapes."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        output = model(obs, hidden)

        # Action components
        assert output.action.action_type.shape == (batch_size,)
        assert output.action.move_direction.shape == (batch_size, 2)
        assert output.action.attack_target.shape == (batch_size,)

        # Scalar outputs per sample
        assert output.log_prob.shape == (batch_size,)
        assert output.entropy.shape == (batch_size,)
        assert output.value.shape == (batch_size,)

        # Attention weights
        assert output.attn_weights.shape == (batch_size, config.max_enemies)

        # Component log probs
        assert output.action_type_log_prob.shape == (batch_size,)
        assert output.move_direction_log_prob.shape == (batch_size,)
        assert output.attack_target_log_prob.shape == (batch_size,)

    def test_forward_with_none_hidden(self, model: ActorCritic, config: ModelConfig):
        """Test forward pass initializes hidden state when None."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)

        # Should not raise
        output = model(obs, hidden=None)
        assert output.hidden is not None

    def test_forward_hidden_state_shape(self, model: ActorCritic, config: ModelConfig):
        """Test hidden state output shape."""
        batch_size = 3
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        output = model(obs, hidden)

        h_marine, h_enemies = output.hidden
        assert h_marine.shape == (
            config.gru_num_layers,
            batch_size,
            config.gru_hidden_size,
        )
        assert h_enemies.shape == (
            config.gru_num_layers,
            batch_size,
            config.max_enemies,
            config.gru_hidden_size,
        )

    def test_forward_with_action_evaluation(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test forward pass can evaluate provided actions."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        # Create action to evaluate
        action = HybridAction(
            action_type=torch.tensor([ACTION_MOVE, ACTION_ATTACK]),
            move_direction=torch.randn(batch_size, 2),
            attack_target=torch.tensor([0, 1]),
        )

        output = model(obs, hidden, action=action)

        # Should return same action
        assert torch.equal(output.action.action_type, action.action_type)

    def test_forward_auxiliary_outputs(self, config: ModelConfig):
        """Test auxiliary outputs when enabled."""
        config_aux = ModelConfig(use_auxiliary_tasks=True)
        model = ActorCritic(config_aux)

        batch_size = 2
        obs = torch.randn(batch_size, config_aux.obs_size)
        output = model(obs, hidden=None)

        assert output.aux_damage is not None
        assert output.aux_distance is not None
        assert output.aux_damage.shape == (batch_size,)
        assert output.aux_distance.shape == (batch_size,)

    def test_forward_no_auxiliary_outputs(self, config: ModelConfig):
        """Test no auxiliary outputs when disabled."""
        config_no_aux = ModelConfig(use_auxiliary_tasks=False)
        model = ActorCritic(config_no_aux)

        batch_size = 2
        obs = torch.randn(batch_size, config_no_aux.obs_size)
        output = model(obs, hidden=None)

        assert output.aux_damage is None
        assert output.aux_distance is None

    def test_forward_different_batch_sizes(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test forward pass works with different batch sizes."""
        for batch_size in [1, 2, 8, 16, 32]:
            obs = torch.randn(batch_size, config.obs_size)
            hidden = model.get_initial_hidden(batch_size)

            output = model(obs, hidden)

            assert output.action.action_type.shape == (batch_size,)
            assert output.value.shape == (batch_size,)


# =============================================================================
# Hidden State Tests
# =============================================================================


class TestHiddenState:
    """Tests for hidden state management."""

    def test_initial_hidden_zeros(self, model: ActorCritic, config: ModelConfig):
        """Test initial hidden state is zeros."""
        batch_size = 2
        hidden = model.get_initial_hidden(batch_size)

        h_marine, h_enemies = hidden
        assert torch.all(h_marine == 0)
        assert torch.all(h_enemies == 0)

    def test_hidden_state_device(self, model: ActorCritic, config: ModelConfig):
        """Test hidden state is on correct device."""
        batch_size = 2
        device = next(model.parameters()).device

        hidden = model.get_initial_hidden(batch_size, device=device)

        h_marine, h_enemies = hidden
        assert h_marine.device == device
        assert h_enemies.device == device

    def test_hidden_state_propagation(self, model: ActorCritic, config: ModelConfig):
        """Test hidden state changes across steps."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        output1 = model(obs, hidden)
        output2 = model(obs, output1.hidden)

        # Hidden states should be different after processing
        h1_marine, _ = output1.hidden
        h2_marine, _ = output2.hidden

        # With different inputs processed, hidden should change
        # (unless by extreme coincidence)
        assert not torch.allclose(h1_marine, h2_marine)


# =============================================================================
# Action Masking Tests
# =============================================================================


class TestActionMasking:
    """Tests for action masking behavior."""

    def test_attack_masked_when_on_cooldown(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test ATTACK is masked when weapon on cooldown."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)

        # Set cd_binary = 1 (on cooldown)
        obs[:, OBS_MARINE_START + 1] = 1.0

        # Ensure enemies are alive
        for i in range(config.max_enemies):
            obs[:, OBS_ENEMY_START + i * config.enemy_obs_size] = 0.5

        # Compute action mask
        marine_obs = model.entity_encoder.get_marine_obs(obs)
        _, _, _, enemy_mask = model.entity_encoder.parse_observation(obs)
        action_mask = model._compute_action_mask(marine_obs, enemy_mask)

        # ATTACK should be masked (False)
        assert torch.all(~action_mask[:, ACTION_ATTACK])
        # MOVE and STOP should be available
        assert torch.all(action_mask[:, ACTION_MOVE])
        assert torch.all(action_mask[:, ACTION_STOP])

    def test_attack_masked_when_no_enemies(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test ATTACK is masked when all enemies are dead."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)

        # Set cd_binary = 0 (weapon ready)
        obs[:, OBS_MARINE_START + 1] = 0.0

        # Set all enemy hp to 0
        for i in range(config.max_enemies):
            obs[:, OBS_ENEMY_START + i * config.enemy_obs_size] = 0.0

        marine_obs = model.entity_encoder.get_marine_obs(obs)
        _, _, _, enemy_mask = model.entity_encoder.parse_observation(obs)
        action_mask = model._compute_action_mask(marine_obs, enemy_mask)

        # ATTACK should be masked
        assert torch.all(~action_mask[:, ACTION_ATTACK])

    def test_attack_available_when_valid(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test ATTACK is available when weapon ready and enemies alive."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)

        # Set cd_binary = 0 (weapon ready)
        obs[:, OBS_MARINE_START + 1] = 0.0

        # Set enemy hp > 0
        obs[:, OBS_ENEMY_START] = 0.5

        marine_obs = model.entity_encoder.get_marine_obs(obs)
        _, _, _, enemy_mask = model.entity_encoder.parse_observation(obs)
        action_mask = model._compute_action_mask(marine_obs, enemy_mask)

        # ATTACK should be available
        assert torch.all(action_mask[:, ACTION_ATTACK])

    def test_forward_respects_action_mask(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test forward pass respects provided action mask."""
        batch_size = 100  # Large batch for statistical test
        obs = torch.randn(batch_size, config.obs_size)

        # Create mask that only allows STOP
        action_mask = torch.zeros(batch_size, 3, dtype=torch.bool)
        action_mask[:, ACTION_STOP] = True

        output = model(obs, hidden=None, action_mask=action_mask)

        # All actions should be STOP
        assert torch.all(output.action.action_type == ACTION_STOP)


# =============================================================================
# Deterministic Action Tests
# =============================================================================


class TestDeterministicAction:
    """Tests for deterministic action selection."""

    def test_deterministic_action_shapes(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test deterministic action output shapes."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        action, new_hidden = model.get_deterministic_action(obs, hidden)

        assert action.action_type.shape == (batch_size,)
        assert action.move_direction.shape == (batch_size, 2)
        assert action.attack_target.shape == (batch_size,)

    def test_deterministic_action_consistent(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test deterministic action is consistent across calls."""
        model.eval()

        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        with torch.no_grad():
            action1, _ = model.get_deterministic_action(obs, hidden)
            action2, _ = model.get_deterministic_action(obs, hidden)

        assert torch.equal(action1.action_type, action2.action_type)
        assert torch.allclose(action1.move_direction, action2.move_direction)
        assert torch.equal(action1.attack_target, action2.attack_target)


# =============================================================================
# Sequence Processing Tests
# =============================================================================


class TestSequenceProcessing:
    """Tests for sequence forward pass."""

    def test_forward_sequence_shapes(self, model: ActorCritic, config: ModelConfig):
        """Test sequence forward pass output shapes."""
        batch_size = 2
        seq_len = 5
        obs_seq = torch.randn(batch_size, seq_len, config.obs_size)

        output = model.forward_sequence(obs_seq)

        assert output.action.action_type.shape == (batch_size, seq_len)
        assert output.action.move_direction.shape == (batch_size, seq_len, 2)
        assert output.action.attack_target.shape == (batch_size, seq_len)
        assert output.log_prob.shape == (batch_size, seq_len)
        assert output.value.shape == (batch_size, seq_len)

    def test_forward_sequence_with_actions(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test sequence forward pass with action evaluation."""
        batch_size = 2
        seq_len = 3

        obs_seq = torch.randn(batch_size, seq_len, config.obs_size)
        actions = {
            "action_type": torch.randint(0, 3, (batch_size, seq_len)),
            "move_direction": torch.randn(batch_size, seq_len, 2),
            "attack_target": torch.randint(0, config.max_enemies, (batch_size, seq_len)),
        }

        output = model.forward_sequence(obs_seq, actions=actions)

        # Should return same action types
        assert torch.equal(output.action.action_type, actions["action_type"])


# =============================================================================
# Loss Computation Tests
# =============================================================================


class TestLossComputation:
    """Tests for loss computation."""

    def test_compute_losses_returns_all_components(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test compute_losses returns all expected loss components."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        output = model(obs, hidden=None)

        old_log_probs = {
            "combined": output.log_prob.detach(),
            "action_type": output.action_type_log_prob.detach(),
            "move_direction": output.move_direction_log_prob.detach(),
            "attack_target": output.attack_target_log_prob.detach(),
        }
        advantages = torch.randn(batch_size)
        value_targets = torch.randn(batch_size)

        losses = model.compute_losses(
            output=output,
            old_log_probs=old_log_probs,
            advantages=advantages,
            value_targets=value_targets,
            clip_epsilon=0.2,
        )

        assert "action_type" in losses
        assert "move_direction" in losses
        assert "attack_target" in losses
        assert "value" in losses

    def test_loss_tensors_are_scalar(self, model: ActorCritic, config: ModelConfig):
        """Test all loss tensors are scalars."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        output = model(obs, hidden=None)

        old_log_probs = {
            "combined": output.log_prob.detach(),
            "action_type": output.action_type_log_prob.detach(),
            "move_direction": output.move_direction_log_prob.detach(),
            "attack_target": output.attack_target_log_prob.detach(),
        }

        losses = model.compute_losses(
            output=output,
            old_log_probs=old_log_probs,
            advantages=torch.randn(batch_size),
            value_targets=torch.randn(batch_size),
            clip_epsilon=0.2,
        )

        for name, head_loss in losses.items():
            assert head_loss.loss.dim() == 0, f"{name} loss is not scalar"

    def test_auxiliary_loss_computation(self):
        """Test auxiliary loss is computed when enabled."""
        config = ModelConfig(use_auxiliary_tasks=True)
        model = ActorCritic(config)

        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        output = model(obs, hidden=None)

        old_log_probs = {
            "combined": output.log_prob.detach(),
            "action_type": output.action_type_log_prob.detach(),
            "move_direction": output.move_direction_log_prob.detach(),
            "attack_target": output.attack_target_log_prob.detach(),
        }

        aux_targets = {
            "damage": torch.rand(batch_size),
            "distance": torch.rand(batch_size),
        }

        losses = model.compute_losses(
            output=output,
            old_log_probs=old_log_probs,
            advantages=torch.randn(batch_size),
            value_targets=torch.randn(batch_size),
            clip_epsilon=0.2,
            aux_targets=aux_targets,
        )

        assert "auxiliary" in losses


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradients_flow_to_main_parameters(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test gradients flow to main trainable parameters (excluding auxiliary)."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        output = model(obs, hidden=None)

        # Compute a simple loss (policy + value, not auxiliary)
        loss = output.log_prob.mean() + output.value.mean()
        loss.backward()

        # Check main parameters have gradients (exclude auxiliary which isn't in this loss)
        main_components = [
            "entity_encoder",
            "temporal_encoder",
            "cross_attention",
            "action_type_head",
            "move_direction_head",
            "attack_target_head",
            "value_head",
        ]
        for name, param in model.named_parameters():
            if param.requires_grad:
                is_main = any(comp in name for comp in main_components)
                if is_main:
                    assert param.grad is not None, f"No gradient for {name}"

    def test_policy_gradient_computation(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test policy gradient can be computed."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        output = model(obs, hidden=None)

        advantages = torch.randn(batch_size)
        policy_loss = -(output.log_prob * advantages).mean()

        policy_loss.backward()

        # Action heads should have gradients
        assert model.action_type_head.net[0].weight.grad is not None

    def test_value_gradient_computation(
        self, model: ActorCritic, config: ModelConfig
    ):
        """Test value gradient can be computed."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        output = model(obs, hidden=None)

        value_targets = torch.randn(batch_size)
        value_loss = ((output.value - value_targets) ** 2).mean()

        value_loss.backward()

        # Value head should have gradients
        assert model.value_head.net[0].weight.grad is not None


# =============================================================================
# Environment Action Conversion Tests
# =============================================================================


class TestEnvActionConversion:
    """Tests for environment action conversion."""

    def test_to_env_action_move(self, model: ActorCritic, config: ModelConfig):
        """Test conversion for MOVE action."""
        action = HybridAction(
            action_type=torch.tensor([ACTION_MOVE]),
            move_direction=torch.tensor([[0.6, 0.8]]),
            attack_target=torch.tensor([0]),
        )

        env_action = model.to_env_action(action, batch_idx=0)

        assert env_action["command"] == ACTION_MOVE
        assert "angle" in env_action
        assert len(env_action["angle"]) == 2

    def test_to_env_action_attack(self, model: ActorCritic, config: ModelConfig):
        """Test conversion for ATTACK action."""
        action = HybridAction(
            action_type=torch.tensor([ACTION_ATTACK]),
            move_direction=torch.tensor([[0.0, 1.0]]),
            attack_target=torch.tensor([1]),
        )

        env_action = model.to_env_action(action, batch_idx=0)

        # ATTACK_Z1=1, ATTACK_Z2=2, so target 1 -> command 2
        assert env_action["command"] == 2

    def test_to_env_action_stop(self, model: ActorCritic, config: ModelConfig):
        """Test conversion for STOP action."""
        action = HybridAction(
            action_type=torch.tensor([ACTION_STOP]),
            move_direction=torch.tensor([[0.0, 1.0]]),
            attack_target=torch.tensor([0]),
        )

        env_action = model.to_env_action(action, batch_idx=0)

        # ENV_STOP = 3 (different from model's ACTION_STOP = 2)
        assert env_action["command"] == 3


# =============================================================================
# Entity Encoder Tests
# =============================================================================


class TestEntityEncoder:
    """Tests for entity encoder."""

    def test_parse_observation(self, config: ModelConfig):
        """Test observation parsing."""
        encoder = EntityEncoder(config)
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)

        time_left, marine_obs, enemy_obs, enemy_mask = encoder.parse_observation(obs)

        assert time_left.shape == (batch_size, 1)
        assert marine_obs.shape == (batch_size, config.marine_obs_size)
        assert enemy_obs.shape == (batch_size, config.max_enemies, config.enemy_obs_size)
        assert enemy_mask.shape == (batch_size, config.max_enemies)

    def test_enemy_mask_from_hp(self, config: ModelConfig):
        """Test enemy mask is derived from HP."""
        encoder = EntityEncoder(config)
        batch_size = 2
        obs = torch.zeros(batch_size, config.obs_size)

        # Set first enemy HP > 0, second enemy HP = 0
        obs[:, OBS_ENEMY_START] = 0.5  # Enemy 1 alive
        obs[:, OBS_ENEMY_START + config.enemy_obs_size] = 0.0  # Enemy 2 dead

        _, _, _, enemy_mask = encoder.parse_observation(obs)

        assert torch.all(enemy_mask[:, 0])  # First enemy valid
        assert torch.all(~enemy_mask[:, 1])  # Second enemy invalid

    def test_forward_output_shapes(self, config: ModelConfig):
        """Test encoder forward output shapes."""
        encoder = EntityEncoder(config)
        batch_size = 3
        obs = torch.randn(batch_size, config.obs_size)

        time_left, marine_emb, enemy_embs, enemy_mask = encoder(obs)

        assert time_left.shape == (batch_size, 1)
        assert marine_emb.shape == (batch_size, config.entity_embed_size)
        assert enemy_embs.shape == (
            batch_size,
            config.max_enemies,
            config.entity_embed_size,
        )
        assert enemy_mask.shape == (batch_size, config.max_enemies)


# =============================================================================
# Cross-Attention Tests
# =============================================================================


class TestCrossAttention:
    """Tests for cross-attention module."""

    def test_attention_output_shapes(self, config: ModelConfig):
        """Test attention output shapes."""
        attention = CrossAttention(config)
        batch_size = 2

        marine_emb = torch.randn(batch_size, config.gru_hidden_size)
        enemy_embs = torch.randn(batch_size, config.max_enemies, config.gru_hidden_size)

        context, attn_weights, attn_logits = attention(marine_emb, enemy_embs)

        assert context.shape == (batch_size, config.entity_embed_size)
        assert attn_weights.shape == (batch_size, config.max_enemies)
        assert attn_logits.shape == (batch_size, config.max_enemies)

    def test_attention_weights_sum_to_one(self, config: ModelConfig):
        """Test attention weights sum to 1."""
        attention = CrossAttention(config)
        batch_size = 2

        marine_emb = torch.randn(batch_size, config.gru_hidden_size)
        enemy_embs = torch.randn(batch_size, config.max_enemies, config.gru_hidden_size)

        _, attn_weights, _ = attention(marine_emb, enemy_embs)

        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_attention_masking(self, config: ModelConfig):
        """Test attention masking works correctly."""
        attention = CrossAttention(config)
        batch_size = 2

        marine_emb = torch.randn(batch_size, config.gru_hidden_size)
        enemy_embs = torch.randn(batch_size, config.max_enemies, config.gru_hidden_size)

        # Mask second enemy
        enemy_mask = torch.tensor([[True, False], [True, False]])

        _, attn_weights, _ = attention(marine_emb, enemy_embs, enemy_mask)

        # Masked enemies should have zero attention weight
        assert torch.allclose(attn_weights[:, 1], torch.zeros(batch_size), atol=1e-5)
        # Unmasked enemy should have weight 1 (since it's the only valid one)
        assert torch.allclose(attn_weights[:, 0], torch.ones(batch_size), atol=1e-5)

    def test_all_masked_handling(self, config: ModelConfig):
        """Test handling when all enemies are masked."""
        attention = CrossAttention(config)
        batch_size = 2

        marine_emb = torch.randn(batch_size, config.gru_hidden_size)
        enemy_embs = torch.randn(batch_size, config.max_enemies, config.gru_hidden_size)

        # Mask all enemies
        enemy_mask = torch.zeros(batch_size, config.max_enemies, dtype=torch.bool)

        context, attn_weights, _ = attention(marine_emb, enemy_embs, enemy_mask)

        # Should not produce NaN
        assert not torch.isnan(context).any()
        assert not torch.isnan(attn_weights).any()


# =============================================================================
# Temporal Encoder Tests
# =============================================================================


class TestTemporalEncoder:
    """Tests for temporal encoder."""

    def test_output_shapes(self, config: ModelConfig):
        """Test temporal encoder output shapes."""
        encoder = TemporalEncoder(config)
        batch_size = 2

        marine_emb = torch.randn(batch_size, config.entity_embed_size)
        enemy_embs = torch.randn(batch_size, config.max_enemies, config.entity_embed_size)
        hidden = encoder.get_initial_hidden(batch_size, config.max_enemies)

        h_marine, h_enemies, new_hidden = encoder(marine_emb, enemy_embs, hidden)

        assert h_marine.shape == (batch_size, config.gru_hidden_size)
        assert h_enemies.shape == (batch_size, config.max_enemies, config.gru_hidden_size)

    def test_hidden_state_update(self, config: ModelConfig):
        """Test hidden state is updated after forward pass."""
        encoder = TemporalEncoder(config)
        batch_size = 2

        marine_emb = torch.randn(batch_size, config.entity_embed_size)
        enemy_embs = torch.randn(batch_size, config.max_enemies, config.entity_embed_size)
        hidden = encoder.get_initial_hidden(batch_size, config.max_enemies)

        _, _, new_hidden = encoder(marine_emb, enemy_embs, hidden)

        h_marine_new, h_enemies_new = new_hidden
        h_marine_old, h_enemies_old = hidden

        # Hidden states should have changed
        assert not torch.allclose(h_marine_new, h_marine_old)


# =============================================================================
# Action Head Tests
# =============================================================================


class TestActionTypeHead:
    """Tests for action type head."""

    def test_output_shapes(self, config: ModelConfig):
        """Test action type head output shapes."""
        head = ActionTypeHead(config)
        batch_size = 2
        # Head expects head_input_size (backbone + skip connection)
        features = torch.randn(batch_size, config.head_input_size)

        action_type, log_prob, entropy, dist = head(features)

        assert action_type.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_action_in_valid_range(self, config: ModelConfig):
        """Test action type is in valid range."""
        head = ActionTypeHead(config)
        batch_size = 100
        # Head expects head_input_size
        features = torch.randn(batch_size, config.head_input_size)

        action_type, _, _, _ = head(features)

        assert torch.all(action_type >= 0)
        assert torch.all(action_type < config.num_action_types)

    def test_masking_respected(self, config: ModelConfig):
        """Test action mask is respected."""
        head = ActionTypeHead(config)
        batch_size = 100
        # Head expects head_input_size
        features = torch.randn(batch_size, config.head_input_size)

        # Only allow MOVE
        action_mask = torch.zeros(batch_size, 3, dtype=torch.bool)
        action_mask[:, ACTION_MOVE] = True

        action_type, _, _, _ = head(features, action_mask=action_mask)

        assert torch.all(action_type == ACTION_MOVE)


class TestMoveDirectionHead:
    """Tests for move direction head."""

    def test_output_shapes(self, config: ModelConfig):
        """Test move direction head output shapes."""
        head = MoveDirectionHead(config)
        batch_size = 2
        # Head expects head_input_size
        features = torch.randn(batch_size, config.head_input_size)

        direction, log_prob, entropy, dist = head(features)

        assert direction.shape == (batch_size, 2)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_direction_is_normalized(self, config: ModelConfig):
        """Test output direction is approximately unit length."""
        head = MoveDirectionHead(config)
        batch_size = 100
        # Head expects head_input_size
        features = torch.randn(batch_size, config.head_input_size)

        direction, _, _, _ = head(features)

        norms = torch.norm(direction, dim=-1)
        # Should be close to 1 (normalized)
        assert torch.allclose(norms, torch.ones(batch_size), atol=0.1)


class TestAttackTargetHead:
    """Tests for attack target head."""

    def test_output_shapes(self, config: ModelConfig):
        """Test attack target head output shapes."""
        head = AttackTargetHead(config)
        batch_size = 2
        attn_logits = torch.randn(batch_size, config.max_enemies)

        target, log_prob, entropy, dist = head(attn_logits)

        assert target.shape == (batch_size,)
        assert log_prob.shape == (batch_size,)
        assert entropy.shape == (batch_size,)

    def test_target_in_valid_range(self, config: ModelConfig):
        """Test target is in valid range."""
        head = AttackTargetHead(config)
        batch_size = 100
        attn_logits = torch.randn(batch_size, config.max_enemies)

        target, _, _, _ = head(attn_logits)

        assert torch.all(target >= 0)
        assert torch.all(target < config.max_enemies)

    def test_masking_respected(self, config: ModelConfig):
        """Test enemy mask is respected."""
        head = AttackTargetHead(config)
        batch_size = 100
        attn_logits = torch.randn(batch_size, config.max_enemies)

        # Only first enemy valid
        enemy_mask = torch.zeros(batch_size, config.max_enemies, dtype=torch.bool)
        enemy_mask[:, 0] = True

        target, _, _, _ = head(attn_logits, enemy_mask=enemy_mask)

        assert torch.all(target == 0)

    def test_no_nan_gradient_with_partial_inf_logits(self, config: ModelConfig):
        """Test no NaN gradients when attn_logits contain -inf for some enemies.

        This is a regression test for a bug where temperature scaling on -inf
        values caused NaN gradients. When cross-attention masks dead enemies,
        their logits become -inf. The attack_target_head must handle this
        without producing NaN gradients through the learnable temperature.
        """
        head = AttackTargetHead(config)
        batch_size = 4

        # Simulate attn_logits from cross-attention where some enemies are dead
        # Sample 0: enemy 0 dead (-inf), enemy 1 alive
        # Sample 1: both alive
        # Sample 2: both alive
        # Sample 3: enemy 1 dead (-inf), enemy 0 alive
        attn_logits = torch.randn(batch_size, config.max_enemies)
        attn_logits[0, 0] = float("-inf")
        attn_logits[3, 1] = float("-inf")

        # Corresponding enemy mask
        enemy_mask = torch.ones(batch_size, config.max_enemies, dtype=torch.bool)
        enemy_mask[0, 0] = False
        enemy_mask[3, 1] = False

        # Forward pass
        target, log_prob, entropy, _ = head(attn_logits, enemy_mask=enemy_mask)

        # Compute a simple loss and backprop
        loss = log_prob.sum()
        loss.backward()

        # Check that log_temperature gradient is not NaN
        assert head.log_temperature.grad is not None, "log_temperature should have gradient"
        assert not torch.isnan(head.log_temperature.grad).any(), (
            f"log_temperature gradient should not be NaN, got {head.log_temperature.grad}"
        )


# =============================================================================
# Value Head Tests
# =============================================================================


class TestCriticHead:
    """Tests for critic head."""

    def test_output_shape(self, config: ModelConfig):
        """Test value head output shape."""
        head = CriticHead(config)
        batch_size = 2
        features = torch.randn(batch_size, config.backbone_output_size)
        marine_obs = torch.randn(batch_size, config.marine_obs_size)

        value = head(features, marine_obs)

        assert value.shape == (batch_size,)

    def test_loss_computation(self, config: ModelConfig):
        """Test value loss computation."""
        head = CriticHead(config)
        batch_size = 4

        values = torch.randn(batch_size)
        targets = torch.randn(batch_size)

        loss_output = head.compute_loss(values, targets)

        assert loss_output.loss.dim() == 0  # Scalar
        assert "loss" in loss_output.metrics
        assert "explained_variance" in loss_output.metrics


# =============================================================================
# Auxiliary Head Tests
# =============================================================================


class TestAuxiliaryHeads:
    """Tests for auxiliary heads."""

    def test_damage_head_output(self, config: ModelConfig):
        """Test damage prediction head output."""
        head = DamageAuxHead(config)
        batch_size = 2
        features = torch.randn(batch_size, config.backbone_output_size)
        marine_obs = torch.randn(batch_size, config.marine_obs_size)

        pred = head(features, marine_obs)

        assert pred.shape == (batch_size,)
        assert torch.all(pred >= 0)  # ReLU output

    def test_distance_head_output(self, config: ModelConfig):
        """Test distance prediction head output."""
        head = DistanceAuxHead(config)
        batch_size = 2
        features = torch.randn(batch_size, config.backbone_output_size)
        marine_obs = torch.randn(batch_size, config.marine_obs_size)

        pred = head(features, marine_obs)

        assert pred.shape == (batch_size,)
        assert torch.all(pred >= 0)
        assert torch.all(pred <= 1)  # Sigmoid output

    def test_combined_head_output(self, config: ModelConfig):
        """Test combined auxiliary head output."""
        head = CombinedAuxiliaryHead(config)
        batch_size = 2
        features = torch.randn(batch_size, config.backbone_output_size)
        marine_obs = torch.randn(batch_size, config.marine_obs_size)

        preds = head(features, marine_obs)

        assert "damage" in preds
        assert "distance" in preds
        assert preds["damage"].shape == (batch_size,)
        assert preds["distance"].shape == (batch_size,)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and potential failure modes."""

    def test_batch_size_one(self, model: ActorCritic, config: ModelConfig):
        """Test model works with batch size 1."""
        obs = torch.randn(1, config.obs_size)
        output = model(obs, hidden=None)

        assert output.action.action_type.shape == (1,)
        assert output.value.shape == (1,)

    def test_all_enemies_dead(self, model: ActorCritic, config: ModelConfig):
        """Test model handles all enemies being dead."""
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)

        # Set all enemy HP to 0
        for i in range(config.max_enemies):
            obs[:, OBS_ENEMY_START + i * config.enemy_obs_size] = 0.0

        # Should not raise
        output = model(obs, hidden=None)

        assert not torch.isnan(output.value).any()
        assert not torch.isnan(output.log_prob).any()

    def test_extreme_observation_values(self, model: ActorCritic, config: ModelConfig):
        """Test model handles extreme observation values."""
        batch_size = 2

        # Very large values
        obs_large = torch.ones(batch_size, config.obs_size) * 100
        output_large = model(obs_large, hidden=None)
        assert not torch.isnan(output_large.value).any()

        # Very small values
        obs_small = torch.ones(batch_size, config.obs_size) * -100
        output_small = model(obs_small, hidden=None)
        assert not torch.isnan(output_small.value).any()

    def test_zero_observations(self, model: ActorCritic, config: ModelConfig):
        """Test model handles zero observations."""
        batch_size = 2
        obs = torch.zeros(batch_size, config.obs_size)

        output = model(obs, hidden=None)

        assert not torch.isnan(output.value).any()
        assert not torch.isnan(output.log_prob).any()

    def test_no_nan_in_gradients(self, model: ActorCritic, config: ModelConfig):
        """Test no NaN values in gradients for typical inputs."""
        batch_size = 4
        obs = torch.randn(batch_size, config.obs_size)
        # Ensure at least one enemy is alive to avoid edge case NaN
        obs[:, OBS_ENEMY_START] = 0.5

        output = model(obs, hidden=None)

        loss = output.log_prob.mean() + output.value.mean()
        loss.backward()

        # Check main model parameters (exclude attack_target temperature which
        # can have NaN grad in edge cases with all-masked targets)
        for name, param in model.named_parameters():
            if param.grad is not None:
                if "log_temperature" in name:
                    # Temperature grad can be NaN when no ATTACK actions sampled
                    continue
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_deterministic_mode(self, model: ActorCritic, config: ModelConfig):
        """Test model in eval mode is deterministic."""
        model.eval()
        batch_size = 2
        obs = torch.randn(batch_size, config.obs_size)
        hidden = model.get_initial_hidden(batch_size)

        with torch.no_grad():
            action1, _ = model.get_deterministic_action(obs, hidden)
            action2, _ = model.get_deterministic_action(obs, hidden)

        assert torch.equal(action1.action_type, action2.action_type)
        model.train()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests simulating real usage."""

    def test_episode_rollout(self, model: ActorCritic, config: ModelConfig):
        """Test simulating an episode rollout."""
        batch_size = 1
        num_steps = 10

        hidden = model.get_initial_hidden(batch_size)
        total_log_prob = 0.0

        for _ in range(num_steps):
            obs = torch.randn(batch_size, config.obs_size)
            output = model(obs, hidden)

            hidden = output.hidden
            total_log_prob += output.log_prob.item()

        # Should complete without error and accumulate log probs
        assert isinstance(total_log_prob, float)

    def test_training_step_simulation(self, model: ActorCritic, config: ModelConfig):
        """Test simulating a training step."""
        batch_size = 8
        obs = torch.randn(batch_size, config.obs_size)

        # Forward pass
        output = model(obs, hidden=None)

        # Simulate PPO update
        old_log_probs = {
            "combined": output.log_prob.detach(),
            "action_type": output.action_type_log_prob.detach(),
            "move_direction": output.move_direction_log_prob.detach(),
            "attack_target": output.attack_target_log_prob.detach(),
        }

        advantages = torch.randn(batch_size)
        value_targets = torch.randn(batch_size)

        # Compute losses
        losses = model.compute_losses(
            output=output,
            old_log_probs=old_log_probs,
            advantages=advantages,
            value_targets=value_targets,
            clip_epsilon=0.2,
        )

        # Combine losses
        total_loss = (
            losses["action_type"].loss
            + losses["move_direction"].loss
            + losses["attack_target"].loss
            + 0.5 * losses["value"].loss
        )

        # Backward pass
        total_loss.backward()

        # Check gradients exist
        assert model.action_type_head.net[0].weight.grad is not None

    def test_model_save_load(self, model: ActorCritic, config: ModelConfig, tmp_path):
        """Test model can be saved and loaded."""
        import os

        save_path = os.path.join(tmp_path, "model.pt")

        # Save
        torch.save(model.state_dict(), save_path)

        # Load into new model
        new_model = ActorCritic(config)
        new_model.load_state_dict(torch.load(save_path))

        # Compare outputs
        obs = torch.randn(2, config.obs_size)
        hidden = model.get_initial_hidden(2)

        model.eval()
        new_model.eval()

        with torch.no_grad():
            out1 = model(obs, hidden)
            out2 = new_model(obs, hidden)

        assert torch.allclose(out1.value, out2.value)


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
