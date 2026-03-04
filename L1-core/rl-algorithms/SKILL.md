---
name: rl-algorithms
description: >
  Production-ready PyTorch scaffolds for 6 deep RL algorithms (DQN, A2C, PPO, DDPG, TD3, SAC) with modular networks, replay/rollout buffers, and Gymnasium-compatible training loops.
layer: L1
domain: [general-rl]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: [rl-methodology]
tags: [pytorch, deep-rl, ppo, sac, td3, ddpg, dqn, a2c, training]
---

# RL Algorithms — Production Deep RL Scaffolds

## Purpose & When to Use

Invoke this skill when you need to:
- **Scaffold a new deep RL algorithm** implementation in PyTorch
- **Select the right algorithm** for a given problem setting
- **Understand practical differences** between algorithms (beyond theory)
- **Implement training loops** compatible with any Gymnasium environment

This skill bridges `rl-methodology` (foundational theory, convergence proofs, tabular/NumPy implementations) and the platform-specific trainers at L2/L3 (rl-tools C++, IsaacLab wrappers, SB3 pipelines).

## Algorithm Selection Guide

| Problem Characteristic | Discrete Actions | Continuous Actions |
|------------------------|------------------|--------------------|
| Simple / low-dimensional | DQN | DDPG |
| Sample efficiency needed | DQN + PER | SAC, TD3 |
| Stability important | Double DQN | TD3, SAC |
| On-policy preferred | A2C | PPO |
| Entropy-regularized | — | SAC |
| Massively parallel envs | — | PPO |
| Real-time / wall-clock budget | DQN | TD3 |

**Quick decision tree:**
1. Discrete actions? → **DQN** (or Double DQN for stability)
2. Continuous actions + on-policy OK? → **PPO** (robust default)
3. Continuous actions + sample efficiency matters? → **SAC** (best off-policy default)
4. Continuous actions + simplicity? → **TD3** (simpler than SAC, still strong)

## Shared Components

### MLP — Configurable Multi-Layer Perceptron

Used by all 6 algorithms for policy and value networks.

```python
import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers and activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### ReplayBuffer — Off-Policy Experience Storage

Used by DQN, DDPG, TD3, SAC.

```python
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor

class ReplayBuffer:
    """Fixed-size circular replay buffer with uniform sampling."""

    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 1_000_000, device: str = "cpu"):
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size
        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int = 256) -> Batch:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return Batch(
            obs=torch.as_tensor(self.obs[idxs], device=self.device),
            actions=torch.as_tensor(self.actions[idxs], device=self.device),
            rewards=torch.as_tensor(self.rewards[idxs], device=self.device),
            next_obs=torch.as_tensor(self.next_obs[idxs], device=self.device),
            dones=torch.as_tensor(self.dones[idxs], device=self.device),
        )
```

### RolloutBuffer — On-Policy Trajectory Storage

Used by A2C, PPO.

```python
import torch
import numpy as np
from dataclasses import dataclass
from typing import Generator

@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor

class RolloutBuffer:
    """Collects on-policy rollouts, computes GAE, yields minibatches."""

    def __init__(self, obs_dim: int, act_dim: int, buffer_size: int, device: str = "cpu"):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.ptr, self.buffer_size = 0, buffer_size
        self.device = device

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        advantages = np.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.buffer_size)):
            next_value = last_value if t == self.buffer_size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        self.advantages = advantages
        self.returns = advantages + self.values

    def get_batches(self, batch_size: int) -> Generator[RolloutBatch, None, None]:
        indices = np.random.permutation(self.buffer_size)
        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start : start + batch_size]
            yield RolloutBatch(
                obs=torch.as_tensor(self.obs[idx], device=self.device),
                actions=torch.as_tensor(self.actions[idx], device=self.device),
                log_probs=torch.as_tensor(self.log_probs[idx], device=self.device),
                returns=torch.as_tensor(self.returns[idx], device=self.device),
                advantages=torch.as_tensor(self.advantages[idx], device=self.device),
                values=torch.as_tensor(self.values[idx], device=self.device),
            )

    def reset(self):
        self.ptr = 0
```

### compute_gae — Generalized Advantage Estimation

Used by A2C, PPO. Standalone version for when you need GAE outside the buffer.

```python
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and discounted returns.

    Returns: (advantages, returns) both shape (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
```

### polyak_update — Soft Target Network Update

Used by DDPG, TD3, SAC.

```python
@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float = 0.005):
    """Soft update: target = tau * source + (1 - tau) * target."""
    for p_src, p_tgt in zip(source.parameters(), target.parameters()):
        p_tgt.data.mul_(1.0 - tau).add_(p_src.data, alpha=tau)
```

### evaluate_policy — Standardized Evaluation Loop

```python
def evaluate_policy(env, policy_fn, n_episodes: int = 10, deterministic: bool = True) -> dict:
    """Evaluate a policy over n episodes. Returns mean/std reward and length.

    Args:
        env: Gymnasium environment
        policy_fn: callable(obs) -> action (numpy)
        n_episodes: number of evaluation episodes
        deterministic: passed to policy if it supports it
    """
    episode_rewards, episode_lengths = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, total_reward, length = False, 0.0, 0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }
```

---

## DQN / Double DQN

### Core Idea

DQN extends tabular Q-learning to high-dimensional spaces using a neural network to approximate Q(s, a). Two key innovations stabilize training: a **replay buffer** decorrelates sequential samples, and a **target network** (updated slowly) prevents moving-target instability. Double DQN decouples action selection from evaluation to reduce overestimation bias. See `rl-methodology` Chapter 8 (Value Function Approximation) for the theoretical foundation.

### Networks

```python
class DQNNet(nn.Module):
    """Q-network: maps obs -> Q-value for each discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.q_net = MLP(obs_dim, n_actions, hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_net(obs)
```

### Update Rule

```python
def dqn_update(
    q_net: DQNNet,
    q_target: DQNNet,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    gamma: float = 0.99,
    double: bool = True,
) -> float:
    """One gradient step. Returns loss value."""
    with torch.no_grad():
        if double:
            # Double DQN: select action with online net, evaluate with target
            next_actions = q_net(batch.next_obs).argmax(dim=1)
            next_q = q_target(batch.next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q = q_target(batch.next_obs).max(dim=1).values
        target = batch.rewards + gamma * (1.0 - batch.dones) * next_q

    current_q = q_net(batch.obs).gather(1, batch.actions.long().unsqueeze(1)).squeeze(1)
    loss = nn.functional.mse_loss(current_q, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()
```

### Complete Training Loop

```python
import gymnasium as gym
import copy

def train_dqn(
    env_id: str = "CartPole-v1",
    total_timesteps: int = 100_000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    learning_starts: int = 1_000,
    target_update_freq: int = 1_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 50_000,
    double: bool = True,
    eval_freq: int = 5_000,
    seed: int = 0,
):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_net = DQNNet(obs_dim, n_actions).to(device)
    q_target = copy.deepcopy(q_net)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(obs_dim, act_dim=1, max_size=buffer_size, device=device)

    obs, _ = env.reset(seed=seed)
    episode_reward, episode_count = 0.0, 0

    for step in range(total_timesteps):
        # Epsilon-greedy exploration
        frac = min(1.0, step / epsilon_decay_steps)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_net(torch.as_tensor(obs, dtype=torch.float32, device=device))
                action = q_values.argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, [action], reward, next_obs, float(terminated))
        obs = next_obs
        episode_reward += reward

        if done:
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0.0

        # Train
        if step >= learning_starts:
            batch = buffer.sample(batch_size)
            dqn_update(q_net, q_target, optimizer, batch, gamma, double)

        # Update target network
        if step % target_update_freq == 0:
            q_target.load_state_dict(q_net.state_dict())

        # Evaluate
        if step % eval_freq == 0:
            def policy_fn(o):
                with torch.no_grad():
                    return q_net(torch.as_tensor(o, dtype=torch.float32, device=device)).argmax().item()
            stats = evaluate_policy(eval_env, policy_fn)
            print(f"Step {step}: mean_reward={stats['mean_reward']:.1f}")

    env.close()
    eval_env.close()
    return q_net
```

### Hyperparameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| lr | 1e-3 | [1e-4, 3e-3] |
| gamma | 0.99 | [0.95, 0.999] |
| buffer_size | 100,000 | [10k, 1M] |
| batch_size | 256 | [32, 512] |
| epsilon_start | 1.0 | — |
| epsilon_end | 0.05 | [0.01, 0.1] |
| epsilon_decay_steps | 50,000 | [10k, 100k] |
| target_update_freq | 1,000 | [500, 10,000] |
| learning_starts | 1,000 | [100, 10,000] |

### Common Pitfalls

- **Overestimation bias**: Use Double DQN (`double=True`). Standard DQN systematically overestimates Q-values.
- **Catastrophic forgetting**: Buffer too small → recent experience overwrites important early transitions. Use ≥100k for most tasks.
- **Learning starts too early**: Training on a near-empty buffer produces high-variance gradients. Wait for ≥1k transitions.
- **Target update too frequent**: Causes oscillation. Hard update every 1k–10k steps, or use soft update (tau=0.005).
- **No gradient clipping**: Large TD errors → exploding gradients. Clip to 10.0.

---

## A2C (Advantage Actor-Critic)

### Core Idea

A2C combines a policy (actor) with a value function (critic) trained on-policy. The critic estimates V(s) and provides a baseline to reduce variance of the policy gradient. The advantage A(s,a) = R - V(s) gives a low-variance, unbiased gradient signal. An entropy bonus encourages exploration. See `rl-methodology` Chapter 10 (Actor-Critic Methods) for convergence analysis of two-timescale updates.

### Networks

```python
class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for discrete or continuous actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256], continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        self.critic = MLP(obs_dim, 1, hidden_dims)
        if continuous:
            self.actor_mean = MLP(obs_dim, act_dim, hidden_dims)
            self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.actor = MLP(obs_dim, act_dim, hidden_dims)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_distribution(self, obs: torch.Tensor):
        if self.continuous:
            mean = self.actor_mean(obs)
            std = self.actor_log_std.exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)
        else:
            logits = self.actor(obs)
            return torch.distributions.Categorical(logits=logits)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        dist = self.get_distribution(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(-1)  # sum over action dims
        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(-1)
        value = self.get_value(obs)
        return action, log_prob, entropy, value
```

### Update Rule

```python
def a2c_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> dict:
    """Single A2C update over a batch of rollout data."""
    _, log_probs, entropy, values = model.get_action_and_value(obs, actions)

    # Policy loss: -E[log_pi * A]
    policy_loss = -(log_probs * advantages.detach()).mean()
    # Value loss: MSE(V, returns)
    value_loss = nn.functional.mse_loss(values, returns)
    # Entropy bonus
    entropy_loss = -entropy.mean()

    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
    }
```

### Complete Training Loop

```python
def train_a2c(
    env_id: str = "CartPole-v1",
    total_timesteps: int = 500_000,
    lr: float = 7e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_steps: int = 5,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_freq: int = 10_000,
    seed: int = 0,
    continuous: bool = False,
):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if continuous else env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorCritic(obs_dim, act_dim, continuous=continuous).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer(obs_dim, act_dim=act_dim if continuous else 1, buffer_size=n_steps, device=device)

    obs, _ = env.reset(seed=seed)
    global_step = 0

    while global_step < total_timesteps:
        buffer.reset()

        # Collect n_steps of experience
        for _ in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.cpu().numpy()
            if not continuous:
                action_np = action_np.item()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            buffer.add(obs, action.cpu().numpy(), reward, float(terminated), log_prob.cpu().numpy(), value.cpu().numpy())
            obs = next_obs
            global_step += 1

            if done:
                obs, _ = env.reset()

        # Compute GAE
        with torch.no_grad():
            last_value = model.get_value(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
        buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

        # Single gradient step over all collected data
        all_data = next(buffer.get_batches(n_steps))
        adv = all_data.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        a2c_update(model, optimizer, all_data.obs, all_data.actions, all_data.returns, adv, entropy_coef, value_coef, max_grad_norm)

        # Evaluate
        if global_step % eval_freq < n_steps:
            def policy_fn(o):
                with torch.no_grad():
                    obs_t = torch.as_tensor(o, dtype=torch.float32, device=device)
                    action, _, _, _ = model.get_action_and_value(obs_t)
                    return action.cpu().numpy() if continuous else action.cpu().item()
            stats = evaluate_policy(eval_env, policy_fn)
            print(f"Step {global_step}: mean_reward={stats['mean_reward']:.1f}")

    env.close()
    eval_env.close()
    return model
```

### Hyperparameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| lr | 7e-4 | [1e-4, 1e-3] |
| gamma | 0.99 | [0.95, 0.999] |
| gae_lambda | 0.95 | [0.9, 1.0] |
| n_steps | 5 | [3, 20] |
| entropy_coef | 0.01 | [0.0, 0.05] |
| value_coef | 0.5 | [0.25, 1.0] |
| max_grad_norm | 0.5 | [0.3, 1.0] |

### Common Pitfalls

- **High variance**: A2C with n_steps=1 has high variance. Use n_steps ≥ 5 or increase to 20 for stability.
- **Entropy collapse**: If entropy drops to zero early, the policy becomes deterministic and stops exploring. Increase `entropy_coef` or use a schedule.
- **Value function lag**: If the critic learns too slowly, advantages are noisy. Increase `value_coef` or use separate learning rates.
- **No advantage normalization**: Raw advantages can have large magnitude; always normalize per-batch.
- **Forgetting to reset buffer**: On-policy data must be discarded after each update. Reusing stale data breaks the on-policy guarantee.

---

## PPO (Proximal Policy Optimization)

### Core Idea

PPO improves on A2C by constraining policy updates to a trust region via a clipped surrogate objective. This prevents destructively large updates that collapse the policy. Multiple minibatch epochs over the same rollout data improve sample efficiency while the clip keeps the policy close to the data-collection policy. See `rl-methodology` Chapter 9 (Policy Gradient) and Chapter 10 (Actor-Critic) for the TRPO motivation and advantage estimation theory.

### Networks

PPO uses the same `ActorCritic` network as A2C (see above).

### Update Rule

```python
def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_buffer: RolloutBuffer,
    n_epochs: int = 10,
    batch_size: int = 64,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    clip_value_loss: bool = True,
) -> dict:
    """PPO update: multiple epochs of minibatch updates over rollout data."""
    total_policy_loss, total_value_loss, total_entropy, n_updates = 0, 0, 0, 0

    for _ in range(n_epochs):
        for batch in rollout_buffer.get_batches(batch_size):
            _, new_log_probs, entropy, new_values = model.get_action_and_value(batch.obs, batch.actions)

            # Advantage normalization
            adv = batch.advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Policy loss with clipping
            ratio = (new_log_probs - batch.log_probs).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (optionally clipped)
            if clip_value_loss:
                v_clipped = batch.values + torch.clamp(new_values - batch.values, -clip_eps, clip_eps)
                v_loss1 = (new_values - batch.returns) ** 2
                v_loss2 = (v_clipped - batch.returns) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                value_loss = 0.5 * nn.functional.mse_loss(new_values, batch.returns)

            entropy_loss = -entropy.mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }
```

### Complete Training Loop

```python
def train_ppo(
    env_id: str = "CartPole-v1",
    total_timesteps: int = 500_000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    n_steps: int = 2048,
    n_epochs: int = 10,
    batch_size: int = 64,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_freq: int = 10_000,
    seed: int = 0,
    continuous: bool = False,
):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0] if continuous else env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorCritic(obs_dim, act_dim, continuous=continuous).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer(obs_dim, act_dim=act_dim if continuous else 1, buffer_size=n_steps, device=device)

    obs, _ = env.reset(seed=seed)
    global_step = 0

    while global_step < total_timesteps:
        buffer.reset()

        # Collect n_steps of experience
        for _ in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.cpu().numpy()
            if not continuous:
                action_np = action_np.item()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            buffer.add(obs, action.cpu().numpy(), reward, float(terminated), log_prob.cpu().numpy(), value.cpu().numpy())
            obs = next_obs
            global_step += 1

            if done:
                obs, _ = env.reset()

        # Compute GAE
        with torch.no_grad():
            last_value = model.get_value(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
        buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

        # PPO update (multiple epochs)
        stats = ppo_update(model, optimizer, buffer, n_epochs, batch_size, clip_eps, entropy_coef, value_coef, max_grad_norm)

        # Evaluate
        if global_step % eval_freq < n_steps:
            def policy_fn(o):
                with torch.no_grad():
                    obs_t = torch.as_tensor(o, dtype=torch.float32, device=device)
                    action, _, _, _ = model.get_action_and_value(obs_t)
                    return action.cpu().numpy() if continuous else action.cpu().item()
            eval_stats = evaluate_policy(eval_env, policy_fn)
            print(f"Step {global_step}: mean_reward={eval_stats['mean_reward']:.1f}, "
                  f"policy_loss={stats['policy_loss']:.4f}, entropy={stats['entropy']:.3f}")

    env.close()
    eval_env.close()
    return model
```

### Hyperparameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| lr | 3e-4 | [1e-4, 1e-3] |
| gamma | 0.99 | [0.95, 0.999] |
| gae_lambda | 0.95 | [0.9, 1.0] |
| n_steps | 2048 | [128, 4096] |
| n_epochs | 10 | [3, 30] |
| batch_size | 64 | [32, 256] |
| clip_eps | 0.2 | [0.1, 0.3] |
| entropy_coef | 0.01 | [0.0, 0.05] |
| value_coef | 0.5 | [0.25, 1.0] |
| max_grad_norm | 0.5 | [0.3, 1.0] |

### Common Pitfalls

- **Too many epochs**: n_epochs > 15 with small clip_eps can still overshoot. Watch for KL divergence spikes.
- **Batch size too large**: If batch_size ≈ n_steps, you get one giant batch per epoch — loses the benefit of minibatch shuffling.
- **No advantage normalization**: Critical for PPO. Without it, the clipping threshold is miscalibrated.
- **Learning rate too high**: PPO is sensitive to LR. Start at 3e-4, reduce if training is unstable.
- **Forgetting to anneal LR**: For long training runs, linearly decaying LR to 0 often helps.

---

## DDPG (Deep Deterministic Policy Gradient)

### Core Idea

DDPG extends DQN to continuous action spaces by learning a deterministic policy mu(s) alongside a Q-function Q(s,a). The actor maximizes Q by backpropagating through the critic, and exploration uses additive Gaussian or Ornstein-Uhlenbeck noise. Both actor and critic use target networks with Polyak averaging. See `rl-methodology` Chapter 9 (Deterministic Policy Gradient theorem) for the theoretical foundation.

### Networks

```python
class DDPGActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256], act_limit: float = 1.0):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_dims, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.act_limit

class DDPGCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)
```

### Update Rule

```python
def ddpg_update(
    actor: DDPGActor,
    critic: DDPGCritic,
    actor_target: DDPGActor,
    critic_target: DDPGCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: Batch,
    gamma: float = 0.99,
    tau: float = 0.005,
) -> dict:
    # Critic update
    with torch.no_grad():
        next_actions = actor_target(batch.next_obs)
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * critic_target(batch.next_obs, next_actions)
    current_q = critic(batch.obs, batch.actions)
    critic_loss = nn.functional.mse_loss(current_q, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor update: maximize Q
    actor_loss = -critic(batch.obs, actor(batch.obs)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft target updates
    polyak_update(actor, actor_target, tau)
    polyak_update(critic, critic_target, tau)

    return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
```

### Complete Training Loop

```python
def train_ddpg(
    env_id: str = "Pendulum-v1",
    total_timesteps: int = 100_000,
    lr_actor: float = 1e-3,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    learning_starts: int = 1_000,
    noise_std: float = 0.1,
    eval_freq: int = 5_000,
    seed: int = 0,
):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = DDPGActor(obs_dim, act_dim, act_limit=act_limit).to(device)
    critic = DDPGCritic(obs_dim, act_dim).to(device)
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    buffer = ReplayBuffer(obs_dim, act_dim, max_size=buffer_size, device=device)

    obs, _ = env.reset(seed=seed)

    for step in range(total_timesteps):
        # Select action with exploration noise
        with torch.no_grad():
            action = actor(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
        action = action + noise_std * np.random.randn(act_dim)
        action = np.clip(action, -act_limit, act_limit)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs if not done else env.reset(seed=None)[0]

        # Train
        if step >= learning_starts:
            batch = buffer.sample(batch_size)
            ddpg_update(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, batch, gamma, tau)

        # Evaluate
        if step % eval_freq == 0:
            def policy_fn(o):
                with torch.no_grad():
                    return actor(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().numpy()
            stats = evaluate_policy(eval_env, policy_fn)
            print(f"Step {step}: mean_reward={stats['mean_reward']:.1f}")

    env.close()
    eval_env.close()
    return actor, critic
```

### Hyperparameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| lr_actor | 1e-3 | [1e-4, 3e-3] |
| lr_critic | 1e-3 | [1e-4, 3e-3] |
| gamma | 0.99 | [0.95, 0.999] |
| tau | 0.005 | [0.001, 0.02] |
| buffer_size | 1,000,000 | [100k, 1M] |
| batch_size | 256 | [64, 512] |
| noise_std | 0.1 | [0.05, 0.3] |
| learning_starts | 1,000 | [100, 25,000] |

### Common Pitfalls

- **Critic-actor coupling**: If the critic is poor, the actor gradient is garbage. Train the critic more (increase lr_critic or use more updates per step).
- **Exploration noise too low**: DDPG with insufficient noise gets stuck in local optima. Start with noise_std=0.1, increase for complex envs.
- **No action clipping**: After adding noise, actions must be clipped to the valid range.
- **Brittle to hyperparameters**: DDPG is notoriously sensitive. Consider TD3 or SAC instead for most tasks.

---

## TD3 (Twin Delayed DDPG)

### Core Idea

TD3 addresses three failure modes of DDPG: (1) overestimation bias via **twin critics** (take the min), (2) high-frequency actor instability via **delayed policy updates**, and (3) target value smoothing via **target policy noise**. See `rl-methodology` Chapter 8 for function approximation error analysis and Chapter 9 for deterministic policy gradient.

### Networks

```python
class TD3Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256], act_limit: float = 1.0):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_dims, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.act_limit

class TD3Critic(nn.Module):
    """Twin Q-networks for TD3."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1)).squeeze(-1)
```

### Update Rule

```python
def td3_update(
    actor: TD3Actor,
    critic: TD3Critic,
    actor_target: TD3Actor,
    critic_target: TD3Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: Batch,
    step: int,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_delay: int = 2,
    target_noise: float = 0.2,
    noise_clip: float = 0.5,
) -> dict:
    # Target policy smoothing
    with torch.no_grad():
        noise = (torch.randn_like(batch.actions) * target_noise).clamp(-noise_clip, noise_clip)
        next_actions = (actor_target(batch.next_obs) + noise).clamp(-actor.act_limit, actor.act_limit)
        target_q1, target_q2 = critic_target(batch.next_obs, next_actions)
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * torch.min(target_q1, target_q2)

    # Critic update
    q1, q2 = critic(batch.obs, batch.actions)
    critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    info = {"critic_loss": critic_loss.item()}

    # Delayed policy update
    if step % policy_delay == 0:
        actor_loss = -critic.q1_forward(batch.obs, actor(batch.obs)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        polyak_update(actor, actor_target, tau)
        polyak_update(critic, critic_target, tau)
        info["actor_loss"] = actor_loss.item()

    return info
```

### Complete Training Loop

```python
def train_td3(
    env_id: str = "Pendulum-v1",
    total_timesteps: int = 100_000,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    learning_starts: int = 25_000,
    policy_delay: int = 2,
    exploration_noise: float = 0.1,
    target_noise: float = 0.2,
    noise_clip: float = 0.5,
    eval_freq: int = 5_000,
    seed: int = 0,
):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = TD3Actor(obs_dim, act_dim, act_limit=act_limit).to(device)
    critic = TD3Critic(obs_dim, act_dim).to(device)
    actor_target = copy.deepcopy(actor)
    critic_target = copy.deepcopy(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    buffer = ReplayBuffer(obs_dim, act_dim, max_size=buffer_size, device=device)

    obs, _ = env.reset(seed=seed)

    for step in range(total_timesteps):
        # Select action with exploration noise
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
            action = action + exploration_noise * np.random.randn(act_dim)
            action = np.clip(action, -act_limit, act_limit)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs if not done else env.reset(seed=None)[0]

        # Train
        if step >= learning_starts:
            batch = buffer.sample(batch_size)
            td3_update(actor, critic, actor_target, critic_target, actor_optimizer, critic_optimizer, batch, step, gamma, tau, policy_delay, target_noise, noise_clip)

        # Evaluate
        if step % eval_freq == 0:
            def policy_fn(o):
                with torch.no_grad():
                    return actor(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().numpy()
            stats = evaluate_policy(eval_env, policy_fn)
            print(f"Step {step}: mean_reward={stats['mean_reward']:.1f}")

    env.close()
    eval_env.close()
    return actor, critic
```

### Hyperparameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| lr_actor | 3e-4 | [1e-4, 1e-3] |
| lr_critic | 3e-4 | [1e-4, 1e-3] |
| gamma | 0.99 | [0.95, 0.999] |
| tau | 0.005 | [0.001, 0.02] |
| buffer_size | 1,000,000 | [100k, 1M] |
| batch_size | 256 | [64, 512] |
| policy_delay | 2 | [2, 3] |
| exploration_noise | 0.1 | [0.05, 0.3] |
| target_noise | 0.2 | [0.1, 0.5] |
| noise_clip | 0.5 | [0.3, 1.0] |
| learning_starts | 25,000 | [1k, 25k] |

### Common Pitfalls

- **Learning starts too early**: TD3 benefits from a full random exploration phase (25k steps). Starting earlier often hurts.
- **Target noise too high**: Makes targets too noisy, hurting convergence. Keep target_noise ≤ 0.5.
- **Policy delay = 1**: Defeats the purpose. Always use delay ≥ 2.
- **Forgetting min(Q1, Q2)**: Using only one critic re-introduces overestimation. Both critics are essential.

---

## SAC (Soft Actor-Critic)

### Core Idea

SAC augments the standard RL objective with an entropy bonus: maximize E[sum(r + alpha * H(pi))]. This encourages exploration, prevents premature convergence, and makes the algorithm robust to hyperparameters. SAC uses the **reparameterization trick** for continuous actions (sample z ~ N(0,1), then a = tanh(mu + sigma * z)), **twin critics** (like TD3), and **automatic temperature tuning** (alpha adjusts to maintain a target entropy). See `rl-methodology` Chapter 9 (policy gradient) and Chapter 10 (actor-critic) for the entropy-regularized objective.

### Networks

```python
class SACCritic(nn.Module):
    """Twin Q-networks for SAC."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class SACActor(nn.Module):
    """Squashed Gaussian policy for SAC."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256], act_limit: float = 1.0):
        super().__init__()
        self.shared = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.shared(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean) * self.act_limit
            return action, torch.zeros(obs.shape[0], device=obs.device)

        # Reparameterization trick
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        # Log-prob with tanh correction
        log_prob = dist.log_prob(z).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)

        return action * self.act_limit, log_prob
```

### Update Rule

```python
def sac_update(
    actor: SACActor,
    critic: SACCritic,
    critic_target: SACCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    batch: Batch,
    gamma: float = 0.99,
    tau: float = 0.005,
    target_entropy: float = -1.0,
) -> dict:
    alpha = log_alpha.exp().item()

    # Critic update
    with torch.no_grad():
        next_actions, next_log_probs = actor(batch.next_obs)
        target_q1, target_q2 = critic_target(batch.next_obs, next_actions)
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * (
            torch.min(target_q1, target_q2) - alpha * next_log_probs
        )
    q1, q2 = critic(batch.obs, batch.actions)
    critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor update
    actions, log_probs = actor(batch.obs)
    q1_pi, q2_pi = critic(batch.obs, actions)
    min_q_pi = torch.min(q1_pi, q2_pi)
    actor_loss = (alpha * log_probs - min_q_pi).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Alpha (temperature) update
    alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # Target update
    polyak_update(critic, critic_target, tau)

    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss.item(),
        "alpha": alpha,
        "alpha_loss": alpha_loss.item(),
    }
```

### Complete Training Loop

```python
def train_sac(
    env_id: str = "Pendulum-v1",
    total_timesteps: int = 100_000,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    lr_alpha: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    learning_starts: int = 5_000,
    target_entropy: float | None = None,
    eval_freq: int = 5_000,
    seed: int = 0,
):
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    if target_entropy is None:
        target_entropy = -float(act_dim)  # heuristic: -dim(A)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = SACActor(obs_dim, act_dim, act_limit=act_limit).to(device)
    critic = SACCritic(obs_dim, act_dim).to(device)
    critic_target = copy.deepcopy(critic)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr_alpha)
    buffer = ReplayBuffer(obs_dim, act_dim, max_size=buffer_size, device=device)

    obs, _ = env.reset(seed=seed)

    for step in range(total_timesteps):
        # Select action (SAC explores via stochastic policy — no external noise)
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action, _ = actor(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs if not done else env.reset(seed=None)[0]

        # Train
        if step >= learning_starts:
            batch = buffer.sample(batch_size)
            sac_update(actor, critic, critic_target, actor_optimizer, critic_optimizer, alpha_optimizer, log_alpha, batch, gamma, tau, target_entropy)

        # Evaluate
        if step % eval_freq == 0:
            def policy_fn(o):
                with torch.no_grad():
                    a, _ = actor(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0), deterministic=True)
                    return a.squeeze(0).cpu().numpy()
            stats = evaluate_policy(eval_env, policy_fn)
            print(f"Step {step}: mean_reward={stats['mean_reward']:.1f}, alpha={log_alpha.exp().item():.3f}")

    env.close()
    eval_env.close()
    return actor, critic
```

### Hyperparameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| lr_actor | 3e-4 | [1e-4, 1e-3] |
| lr_critic | 3e-4 | [1e-4, 1e-3] |
| lr_alpha | 3e-4 | [1e-4, 1e-3] |
| gamma | 0.99 | [0.95, 0.999] |
| tau | 0.005 | [0.001, 0.02] |
| buffer_size | 1,000,000 | [100k, 1M] |
| batch_size | 256 | [64, 512] |
| target_entropy | -dim(A) | [-2*dim(A), -0.5*dim(A)] |
| learning_starts | 5,000 | [1k, 25k] |

### Common Pitfalls

- **Fixed alpha**: Using a constant temperature loses SAC's main advantage. Always use auto-tuning unless you have a specific reason.
- **Wrong target entropy**: The heuristic -dim(A) works well in practice. Too low → over-exploration; too high → under-exploration.
- **Log-prob numerical issues**: The tanh squashing correction `log(1 - tanh(x)^2)` needs a small epsilon (1e-6) to avoid log(0).
- **No target network for critic**: Unlike the actor, the critic needs a target network. Using the live critic for bootstrap targets causes divergence.
- **Sharing critic params with actor**: SAC requires separate networks. The actor loss backprops through the critic but only updates the actor.

---

## Hyperparameter Quick Reference

Consolidated defaults across all algorithms:

| Param | DQN | A2C | PPO | DDPG | TD3 | SAC |
|-------|-----|-----|-----|------|-----|-----|
| lr (actor) | 1e-3 | 7e-4 | 3e-4 | 1e-3 | 3e-4 | 3e-4 |
| lr (critic) | — | — | — | 1e-3 | 3e-4 | 3e-4 |
| gamma | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| tau | — | — | — | 0.005 | 0.005 | 0.005 |
| buffer_size | 100k | — | — | 1M | 1M | 1M |
| batch_size | 256 | n_steps | 64 | 256 | 256 | 256 |
| n_steps | — | 5 | 2048 | — | — | — |
| n_epochs | — | 1 | 10 | — | — | — |
| clip_eps | — | — | 0.2 | — | — | — |
| entropy_coef | — | 0.01 | 0.01 | — | — | auto |
| learning_starts | 1k | — | — | 1k | 25k | 5k |
| exploration | eps-greedy | entropy | entropy | Gaussian | Gaussian | stochastic |
| action type | discrete | both | both | continuous | continuous | continuous |
| on/off-policy | off | on | on | off | off | off |
