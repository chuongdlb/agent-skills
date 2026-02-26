# RL Training Configs

Full attribute tables for all RL library configurations.

## RSL-RL Configs

### RslRlPpoActorCriticRecurrentCfg

Extends `RslRlPpoActorCriticCfg` for recurrent policies:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_name` | str | "ActorCriticRecurrent" | Policy class |
| `rnn_type` | str | MISSING | "lstm" or "gru" |
| `rnn_hidden_dim` | int | MISSING | Hidden dimension |
| `rnn_num_layers` | int | MISSING | Number of RNN layers |

### RslRlSymmetryCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_data_augmentation` | bool | False | Augment data with symmetry |
| `use_mirror_loss` | bool | False | Mirror symmetry loss |
| `data_augmentation_func` | Callable | MISSING | Symmetry function |
| `mirror_loss_coeff` | float | 0.0 | Mirror loss coefficient |

### RslRlRndCfg (Random Network Distillation)

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight` | float | 0.0 | RND reward weight |
| `reward_normalization` | bool | False | Normalize RND reward |
| `state_normalization` | bool | False | Normalize RND input |
| `learning_rate` | float | 1e-3 | Predictor learning rate |
| `num_outputs` | int | 1 | Output dimensions |
| `predictor_hidden_dims` | list[int] | [-1] | Predictor MLP |
| `target_hidden_dims` | list[int] | [-1] | Target MLP |

### RslRlDistillationRunnerCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_name` | str | "DistillationRunner" | Runner class |
| `policy` | RslRlDistillationStudentTeacherCfg | MISSING | Student-teacher config |
| `algorithm` | RslRlDistillationAlgorithmCfg | MISSING | Distillation algorithm |

### RslRlDistillationStudentTeacherCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_name` | str | "StudentTeacher" | Policy class |
| `init_noise_std` | float | MISSING | Initial noise |
| `student_hidden_dims` | list[int] | MISSING | Student MLP |
| `teacher_hidden_dims` | list[int] | MISSING | Teacher MLP |
| `activation` | str | MISSING | Activation function |
| `student_obs_normalization` | bool | MISSING | Normalize student obs |
| `teacher_obs_normalization` | bool | MISSING | Normalize teacher obs |

### RslRlDistillationAlgorithmCfg

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_name` | str | "Distillation" | Algorithm class |
| `num_learning_epochs` | int | MISSING | Epochs per iteration |
| `learning_rate` | float | MISSING | Learning rate |
| `gradient_length` | int | MISSING | BPTT gradient length |
| `max_grad_norm` | float \| None | None | Gradient clipping |
| `optimizer` | str | "adam" | "adam", "adamw", "sgd", "rmsprop" |
| `loss_type` | str | "mse" | "mse" or "huber" |

## RL-Games Config (YAML Format)

```yaml
params:
  seed: 42
  algo:
    name: a2c_continuous
  model:
    name: continuous_a2c_logstd
  network:
    name: actor_critic
    separate: False
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        fixed_sigma: True
    mlp:
      units: [256, 128, 64]
      activation: elu
  config:
    name: MyTask
    env_name: rlgpu
    score_to_win: 20000
    normalize_advantage: True
    gamma: 0.99
    tau: 0.95
    learning_rate: 3e-4
    lr_schedule: adaptive
    kl_threshold: 0.008
    mini_epochs: 5
    minibatch_size: 32768
    num_actors: -1
    horizon_length: 16
    max_epochs: 1000
    save_best_after: 100
    save_frequency: 50
```

### RL-Games PBT Config

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable PBT |
| `num_policies` | int | 8 | Population size |
| `interval_steps` | int | 100000 | Steps between evolution |
| `mutation_rate` | float | 0.25 | Mutation probability |
| `change_range` | tuple | (1.1, 2.0) | Mutation range |
| `objective` | str | "Episode_Reward/success" | Optimization target |

## SB3 Config (YAML Format)

```yaml
seed: 42
n_timesteps: 36000
policy: MlpPolicy
n_steps: 24
batch_size: 24576
n_epochs: 5
learning_rate: 0.001
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
vf_coef: 0.5
ent_coef: 0.01
max_grad_norm: 1.0
policy_kwargs:
  activation_fn: nn.ELU
  net_arch:
    pi: [256, 128, 64]
    vf: [256, 128, 64]
```

## SKRL Config (YAML Format)

```yaml
seed: 42
models:
  separate: False
  policy:
    class: GaussianMixin
    clip_actions: False
    clip_log_std: True
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
      - name: net
        input: OBSERVATIONS
        layers: [256, 128, 64]
        activation: elu
agent:
  class: PPO
  rollouts: 24
  learning_epochs: 5
  mini_batches: 4
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 1.0e-03
  learning_rate_scheduler: KLAdaptiveLR
  state_preprocessor: RunningStandardScaler
  value_preprocessor: RunningStandardScaler
  entropy_loss_scale: 0.01
  value_loss_scale: 1.0
  clip_predicted_values: True
  rewards_shaper: null
trainer:
  class: SequentialTrainer
  timesteps: 36000
  environment_info: log
```

## Wrapper Comparison

| Feature | RSL-RL | RL-Games | SB3 | SKRL |
|---------|--------|----------|-----|------|
| GPU tensors | Yes | Yes | No (numpy) | Yes |
| Multi-agent | Via conversion | No | No | Yes (MAPPO) |
| Asymmetric obs | Yes | Yes | Yes | Yes |
| Action clipping | In wrapper | In wrapper | In wrapper | In wrapper |
| Distributed | Yes | Yes | No | Yes |
| Config format | Python | YAML | YAML | YAML |
| JAX support | No | No | No | Yes |
