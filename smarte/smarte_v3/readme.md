## Comprehensive Model Architecture Report: SMARTE v3

### 1. Overall Model Architecture

**Architecture Style: AlphaZero-style Parallel Prediction**

The SMARTE v3 model implements an **AlphaZero-style actor-critic architecture** with independent parallel prediction heads for hybrid action space (discrete + continuous). Key characteristics:

```
Observation (B, obs_size) 
    ↓
    ├─→ CommandHead (Discrete) → command action
    ├─→ AngleHead (Continuous) → angle action
    └─→ CriticHead → value estimate
```

**Key Architectural Philosophy:**
- **No action autoregression**: All heads predict independently from observations (not conditioned on command)
- **Masking at loss time**: Invalid actions are masked at loss computation, not during forward pass
- **Independent probability factorization**: P(action | obs) = P(command | obs) × P(angle | obs)
- **Parallel prediction enables**: Clean PPO gradient flow without action-dependent conditioning complexity

**File Location:** `smarte/smarte_v3/model/actor_critic.py`

---

### 2. Observation Structure and Processing

**Observation Specification: `smarte/smarte_v3/obs.py`**

The `ObsSpec` class is the **single source of truth** for observation layout. It defines:

#### Observation Composition:
```python
ObsSpec(
    num_allies=1,        # Single Marine
    num_enemies=2,       # Two Zerglings
    game_state_size=1,   # Time remaining
    ally_feature_size=5,
    enemy_feature_size=7
)
```

**Total Observation Size:** 1 + (1×5) + (2×7) = **20 features**

#### Detailed Feature Breakdown:

**Game State (1 feature):**
- `time_remaining`: Normalized [0, 1]

**Ally Features per Unit (5 features):**
1. `health`: Normalized [0, 1]
2. `weapon_cooldown`: Binary (0 or 1)
3. `cooldown_norm`: Normalized [0, 1] (capped at max_cooldown=15.0)
4. `facing_sin`: sin(facing_angle) - unit facing direction
5. `facing_cos`: cos(facing_angle)

**Enemy Features per Unit (7 features):**
1. `health`: Normalized [0, 1]
2. `angle_sin`: sin(angle_to_enemy) - direction from reference ally to enemy
3. `angle_cos`: cos(angle_to_enemy)
4. `distance_norm`: Normalized [0, 1] (capped at max_distance=30.0)
5. `in_attack_range`: Binary (0 or 1) - whether enemy is within reference unit's attack range
6. `facing_sin`: sin(enemy_facing_angle)
7. `facing_cos`: cos(enemy_facing_angle)

#### Observation Building Process:
```python
def build(time_remaining, allies, enemies):
    obs = np.zeros(total_size)
    obs[game_state_slice] = [time_remaining]
    
    # Encode allies (reference is first ally)
    for i, ally_slice in ally_slices:
        obs[ally_slice] = _encode_ally(allies[i])
    
    # Encode enemies RELATIVE to reference ally
    for i, enemy_slice in enemy_slices:
        obs[enemy_slice] = _encode_enemy(reference, enemies[i])
    
    return obs
```

**Key Design Decisions:**
- **Relative encoding**: Enemy angles/distances are relative to the reference (first) ally
- **Angle representation**: Uses (sin, cos) pairs instead of raw angles for continuity
- **Fixed structure**: Always 2 enemies; missing/dead enemies are zero-encoded
- **Normalization**: Distance and cooldown have explicit max values for stable gradients

---

### 3. Enemy Representation in Observation Space

**How Enemies Flow Through the Network:**

#### Data Flow:
```
Raw Enemy State (from SC2 API)
    ↓
    Relative to Reference Marine:
    - angle_to(enemy) → (sin, cos)
    - distance_to(enemy) → normalized [0, 1]
    - in_attack_range → binary
    - facing → (sin, cos)
    ↓
Flattened into Observation Vector
    (indices 6-12 for Zergling #1, 13-19 for Zergling #2)
    ↓
Shared by All Heads:
    CommandHead uses enemy features for command selection
    AngleHead uses enemy features for angle prediction (auxiliary task)
    CriticHead uses enemy features for value estimation
    ↓
Auxiliary Prediction Task (AngleHead):
    Predicts all ally+enemy features from hidden state
    Forces encoder to represent enemy state
```

#### Specific Enemy Features Used:
1. **Health**: Critical for decision-making (prioritize low-health targets)
2. **Angle & Distance**: Essential for navigation and attack range decisions
3. **In-Range Binary**: Action validity (can only attack if in range)
4. **Enemy Facing**: Predictive of movement direction (auxiliary task target)

#### Attack Range Computation:
```python
UNIT_ATTACK_RANGE = {
    UNIT_MARINE: 5.0 + 1.0,      # 6.0
    UNIT_ZERGLING: 0.0 + 1.0,    # 1.0 (melee)
}

# Action mask for attacks:
mask[ATTACK_Z1] = (enemy_alive and weapon_ready and distance < marine.attack_range)
```

---

### 4. Head Architectures and Attention-like Mechanisms

#### **CommandHead: Discrete Command Selection**
**File:** `smarte/smarte_v3/model/heads/command.py`

```python
class CommandHead(ActionHead):
    net = nn.Sequential(
        nn.Linear(obs_size, head_hidden_size),  # 20 → 32
        nn.Tanh(),
        nn.Linear(head_hidden_size, num_commands)  # 32 → 3
    )
```

**Processing:**
1. Forward pass produces logits for [MOVE, ATTACK_Z1, ATTACK_Z2]
2. **Action Mask Application**: Sets invalid commands to -∞
3. **Distribution**: Categorical over valid actions
4. **Loss**: PPO-clipped policy gradient

**Masking Logic:**
```python
logits = logits.masked_fill(~mask, float("-inf"))
dist = Categorical(logits=logits)
```

---

#### **AngleHead: Continuous Movement Direction (Von Mises Distribution)**
**File:** `smarte/smarte_v3/model/heads/angle.py`

This is the most sophisticated head, using the von Mises distribution (circular Gaussian).

```python
class AngleHead(ActionHead):
    # Deeper encoder for better feature extraction
    encoder = nn.Sequential(
        [nn.Linear(obs_size, head_hidden_size) + nn.SiLU() 
         for _ in range(angle_encoder_layers)]  # Default: 2 layers
    )
    
    # Output head maps hidden → angle in radians
    output_head = nn.Sequential(
        [nn.Linear(head_hidden_size, head_hidden_size) + nn.SiLU() 
         for _ in range(angle_output_layers-1)],  # Default: 1 hidden layer
        nn.Linear(head_hidden_size, 1)  # Single output: θ
    )
    
    # Learnable concentration parameter (exploration)
    log_concentration = Parameter(init_log_concentration)
    
    # Auxiliary prediction head (forces representation learning)
    aux_head = nn.Sequential(
        nn.Linear(head_hidden_size, aux_hidden_size),
        nn.SiLU(),
        nn.Linear(aux_hidden_size, aux_target_size)  # Predicts all ally+enemy features
    )
```

**Why Von Mises Instead of Gaussian?**

The problem with Gaussian on (sin, cos):
- Gaussian noise on (sin, cos) gives poor angular coverage
- With mean=(0, 7) and std=1.8, policy can only explore ~20° around current direction
- **Von Mises samples angles directly**, giving uniform angular exploration regardless of concentration

**Von Mises Distribution:**
- **Defined on circle [0, 2π)** with natural wrap-around
- **Concentration parameter κ (kappa)**:
  - κ → 0: uniform distribution (maximum exploration)
  - κ ≈ 1: moderate concentration (std ≈ 65°)
  - κ ≈ 2: tighter (std ≈ 45°)
  - κ ≈ 4: fairly concentrated (std ≈ 30°)
  - κ → ∞: point mass (no exploration)

**Entropy Computation (Von Mises):**
```python
def von_mises_entropy(concentration):
    # H = log(2π) + log(I_0(κ)) - κ * I_1(κ) / I_0(κ)
    # Where I_0, I_1 are modified Bessel functions
    i0e = torch.special.i0e(concentration)
    i1e = torch.special.i1e(concentration)
    log_i0 = torch.log(i0e) + concentration
    entropy = log(2π) + log_i0 - concentration * (i1e / i0e)
    return entropy
```

**Action Processing Flow:**
```python
h = encoder(obs)                          # (B, hidden_size)
theta_mean = output_head(h).squeeze(-1)   # (B,) - angle in radians
concentration = softplus(log_concentration) # scalar, clamped [0.01, 100]
dist = VonMises(theta_mean, concentration)

if action is None:
    theta_sample = dist.sample()  # Sample from von Mises
    action = [sin(theta_sample), cos(theta_sample)]  # Convert to environment format
    log_prob = dist.log_prob(theta_sample)
else:
    # Evaluate provided (sin, cos) action
    theta_action = atan2(sin, cos)
    log_prob = dist.log_prob(theta_action)
```

**Auxiliary Prediction Task:**
```python
def compute_aux_loss(obs):
    h = encoder(obs)
    aux_pred = aux_head(h)  # Predict all ally+enemy features
    aux_targets = concatenate([obs[:, s] for s in aux_target_slices])
    return MSE(aux_pred, aux_targets)
```

**Why Auxiliary Task?**
- Forces encoder to represent observation features (enemy angles, distances)
- Prevents encoder collapse where all observations map to similar hidden states
- Without it, policy gradients cancel across episodes with different optimal actions
- Acts as supervised learning signal that doesn't average across episodes

---

#### **CriticHead: State Value Estimation**
**File:** `smarte/smarte_v3/model/heads/value.py`

```python
class CriticHead(nn.Module):
    net = nn.Sequential(
        nn.Linear(obs_size, head_hidden_size),  # 20 → 32
        nn.Tanh(),
        nn.Linear(head_hidden_size, 1)          # 32 → 1 (scalar value)
    )
    
    def compute_loss(values, targets, clip_epsilon=None, old_values=None):
        if clip_epsilon and old_values:
            # PPO-style clipped value loss
            value_clipped = old_values + clamp(values - old_values, -ε, ε)
            loss_unclipped = smooth_l1_loss(values, targets)
            loss_clipped = smooth_l1_loss(value_clipped, targets)
            loss = max(loss_unclipped, loss_clipped).mean()
        else:
            loss = smooth_l1_loss(values, targets)
        return loss
```

---

### 5. Variable-Length Input Handling

**How the System Handles Variable Number of Enemies:**

#### Current Design (Fixed Structure):
```python
ObsSpec(num_enemies=2)  # Always expects exactly 2 enemies
```

- **Fixed enemy slots**: Always allocates space for 2 enemies
- **Zero-encoding for missing**: If an enemy dies, its feature vector becomes all zeros
- **No masking needed**: The observation encoder naturally handles zero padding

#### In the Observation Building:
```python
for i, enemy_slice in enumerate(self.enemy_slices):  # Iterates over [0, 1]
    if i < len(enemies) and reference is not None:
        obs[enemy_slice] = _encode_enemy(reference, enemies[i])
    # else: remains zero (dead/missing enemy)
```

#### At Episode Collection Time:
```python
def collect_episode(env, model, ...):
    obs, info = env.reset()
    obs_tensor = torch.empty(1, obs.shape[0])  # Pre-allocated
    
    while not done:
        obs_tensor[0].copy_(torch.from_numpy(obs))  # Copy into pre-allocated
        mask_tensor = torch.from_numpy(action_mask).unsqueeze(0)
        output = model(obs_tensor, action_mask=mask_tensor)
        # Model processes variable-length obs without issues
```

#### Model Flexibility:
- **Fully dense computation**: All observations flow through dense layers
- **No sequential processing**: No attention or RNN; all features processed in parallel
- **Graceful degradation**: Zero enemy features naturally suppress relevance
- **No need for attention masking**: The model learns implicitly from the zero-padding

---

### 6. Training Flow and Loss Computation

**File:** `smarte/smarte_v3/actor_critic.py`

#### Forward Pass:
```python
def forward(obs, command=None, angle=None, *, action_mask):
    # All heads predict independently
    command_output = command_head(obs, action=command, mask=action_mask)
    angle_output = angle_head(obs, action=angle)
    value = value_head(obs)
    return ActorCriticOutput(command, angle, value)
```

#### Loss Computation:
```python
def compute_losses(output, old_cmd_log_prob, old_angle_log_prob, advantages, vtrace_targets, clip_epsilon):
    cmd_loss = command_head.compute_loss(
        new_log_prob=output.command.log_prob,
        old_log_prob=old_cmd_log_prob,
        advantages=advantages,
        clip_epsilon=clip_epsilon
    )
    
    angle_loss = angle_head.compute_loss(
        new_log_prob=output.angle.log_prob,
        old_log_prob=old_angle_log_prob,
        advantages=advantages,
        clip_epsilon=clip_epsilon
    )
    
    value_loss = value_head.compute_loss(values=output.value, targets=vtrace_targets)
    
    # Auxiliary task
    aux_loss = angle_head.compute_aux_loss(obs)
    
    return {
        "command": cmd_loss,
        "angle": angle_loss,
        "value": value_loss,
        "aux": aux_loss
    }
```

#### PPO Loss Formula (Applied to Both Heads):
```
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
loss = -min(surr1, surr2).mean()
```

#### Total Training Loss:
```python
total_loss = cmd_loss + angle_loss + value_coef * value_loss + aux_coef * aux_loss
```

**From config.py:**
```python
entropy_coef: float = 0.05      # High entropy for exploration
value_coef: float = 0.5          # Value function weight
aux_coef: float = 0.5            # Auxiliary task weight (equal to value)
```

---

### 7. Information Flow: Enemy Data Path Through Network

**Complete Enemy Information Flow:**

```
Raw SC2 Game State (Zerglings)
├─ Position (x, y)
├─ Health / Health_max
├─ Facing angle
├─ Weapon cooldown
└─ Attack range

        ↓ _encode_enemy(reference, enemy)

Relative Features (7 per enemy)
├─ health / health_max                         [0, 1]
├─ angle_sin = sin(angle_to_enemy)             [-1, 1]
├─ angle_cos = cos(angle_to_enemy)             [-1, 1]
├─ distance_norm = dist / max_dist             [0, 1]
├─ in_attack_range (binary)                    {0, 1}
├─ facing_sin = sin(enemy.facing)              [-1, 1]
└─ facing_cos = cos(enemy.facing)              [-1, 1]

        ↓ obs[enemy_slices] = encoded_features

Observation Vector (indices 6-19 for 2 enemies)
├─ game_state[0:1]                  1 feature
├─ ally_features[1:6]               5 features
├─ enemy_0_features[6:13]           7 features
└─ enemy_1_features[13:20]          7 features

        ↓ model(obs, action_mask)

CommandHead (Dense 20→32→3)
├─ Processes all features including enemy data
├─ Outputs: [P(MOVE), P(ATTACK_Z1), P(ATTACK_Z2)]
├─ Action mask applied: sets P(invalid_action) = 0
└─ Attack_Z1/Z2 only valid if: in_range AND weapon_ready AND alive

AngleHead Encoder (Dense 20→32→32 with SiLU)
├─ Extracts hidden representation from all features
├─ Hidden state captures enemy relative position and state
├─ Output: mean angle θ for von Mises distribution
├─ Entropy/Concentration: learned global parameter κ
└─ Action: (sin θ, cos θ) for environment

AngleHead Auxiliary Task (32→16→20)
├─ Predicts all observation features from hidden state
├─ Targets: [ally_health, ally_cooldown_norm, ally_facing_sin/cos,
│            enemy_0_health, enemy_0_angle_sin/cos, ...,
│            enemy_1_health, enemy_1_angle_sin/cos, ...]
├─ Loss: MSE(predicted_features, actual_features)
└─ Effect: Forces encoder to maintain enemy state representation

CriticHead (Dense 20→32→1)
├─ Processes all features for value estimation
├─ Outputs: V(state) - expected cumulative reward
└─ Used for: advantage computation, bootstrap

        ↓ compute_losses()

PPO Policy Loss = -min(ratio*adv, clamp(ratio)*adv)
  where ratio = exp(π_new / π_old)
  
Value Loss = smooth_l1(V_predicted, V_target)
  where V_target from GAE (generalized advantage estimation)

Auxiliary Loss = MSE(predicted_features, actual_features)
  penalizes when encoder fails to represent features
```

---

### 8. Gradient Flow and Representation Learning

**How Enemy Data Influences Gradients:**

1. **CommandHead Path:**
   - Enemy in_range features directly influence attack command logits
   - Gradients flow backward through all 20 input features
   - Attack impossibility (dead enemy) → gradient on in_range → gradient on distance → position encoding

2. **AngleHead Path:**
   - Enemy angle features determine optimal movement angle
   - Multi-layer encoder learns compressed representation
   - Auxiliary task supervises: "you must predict enemy angle from your hidden state"
   - This prevents encoder from collapsing enemy information

3. **Value Head Path:**
   - Enemy health heavily influences expected returns
   - Critic learns: "low enemy health → high value", "high player damage taken → low value"
   - These gradients help other heads learn similar representations

4. **Auxiliary Task (Key Innovation):**
   ```python
   # Without auxiliary task:
   # If both enemies die, all episodes get negative reward
   # Policy gradients can cancel across episodes
   # Encoder could ignore enemy features entirely
   
   # With auxiliary task:
   # Explicit supervision: "predict enemy health/angle from hidden state"
   # Loss doesn't average across episodes
   # Encoder must maintain enemy-specific information
   ```

---

### 9. Summary Table: Architecture Components

| Component | Input | Output | Hidden Layers | Key Features |
|-----------|-------|--------|---|---|
| **CommandHead** | 20 (obs) | 3 (logits) | 32 | Categorical, action masking |
| **AngleHead Encoder** | 20 (obs) | 32 | SiLU, 2 layers | Compresses all features |
| **AngleHead Output** | 32 (hidden) | 1 (angle) | None | Von Mises mean |
| **AngleHead Aux** | 32 (hidden) | 20 (features) | 16 | MSE supervision |
| **CriticHead** | 20 (obs) | 1 (value) | 32 | Tanh, smooth L1 loss |
| **Total Params** | - | - | ~12-15K | Small, efficient |

---

### 10. Key Design Insights

1. **No Attention Mechanisms**: The architecture doesn't use attention. Instead, it relies on:
   - **Dense mixing**: All features processed through fully-connected layers
   - **Implicit importance weighting**: Network learns which features matter
   - **Auxiliary supervision**: Forces certain representations

2. **Variable Enemy Handling**: 
   - Fixed structure (always 2 slots)
   - Zero-padding for missing/dead enemies
   - Network naturally learns to ignore zero features
   - No explicit masking needed (unlike transformer-based approaches)

3. **Enemy Information Channels**:
   - **Health** → criticality (which to target)
   - **Angle/Distance** → navigation and attack decisions
   - **In-Range** → action validity
   - **Facing** → movement prediction (auxiliary task)

4. **Exploration via Von Mises**:
   - Better than Gaussian on (sin, cos)
   - Learned concentration parameter κ for adaptive exploration
   - Uniform angular coverage at all concentrations

This architecture achieves strong performance through simplicity and targeted supervision rather than complex attention mechanisms.
