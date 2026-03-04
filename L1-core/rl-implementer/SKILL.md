---
name: rl-implementer
description: >
  Implements RL algorithms with correct mathematical grounding, translating from theoretical framework to working Python code with grid worlds, learning rate schedules, and DQN.
layer: L1
domain: [general-rl]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: [rl-theory-analyzer]
tags: [implementation, python, tabular, dqn, grid-world]
---

# RL Implementer

## Purpose
Implement RL algorithms with correct mathematical grounding, translating from the theoretical framework in Zhao's book to working Python code.

## When to Use
Invoke this skill when you need to:
- Implement an RL algorithm from pseudocode
- Set up grid world environments matching the book's examples
- Implement proper learning rate schedules satisfying SA conditions
- Add experience replay, target networks, or other stabilization techniques
- Debug an RL implementation by checking mathematical correctness

## Implementation Framework

### Environment Setup (Grid World)
The book uses a 3×3 grid world throughout. Standard implementation:

```python
import numpy as np
from typing import Tuple, Dict, List, Optional

class GridWorld:
    """3×3 grid world environment from Zhao's book."""

    def __init__(self, size: int = 3, gamma: float = 0.9):
        self.size = size
        self.n_states = size * size
        self.n_actions = 5  # up, down, left, right, stay
        self.gamma = gamma
        # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_effects = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)
        }
        self.forbidden_states = set()  # walls/obstacles
        self.target_state = None       # goal state
        self.rewards = {}              # (s, a, s') -> r

    def state_to_rc(self, s: int) -> Tuple[int, int]:
        return s // self.size, s % self.size

    def rc_to_state(self, r: int, c: int) -> int:
        return r * self.size + c

    def step(self, state: int, action: int) -> Tuple[int, float]:
        r, c = self.state_to_rc(state)
        dr, dc = self.action_effects[action]
        nr, nc = r + dr, c + dc
        # Boundary check
        if 0 <= nr < self.size and 0 <= nc < self.size:
            next_state = self.rc_to_state(nr, nc)
            if next_state not in self.forbidden_states:
                reward = self.rewards.get((state, action, next_state), -1)
                return next_state, reward
        # Stay in place if invalid move
        reward = self.rewards.get((state, action, state), -1)
        return state, reward

    def get_transition_model(self) -> np.ndarray:
        """Returns p(s'|s,a) as array of shape (n_states, n_actions, n_states)."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                s_next, _ = self.step(s, a)
                P[s, a, s_next] = 1.0  # deterministic
        return P
```

### Algorithm Templates

#### Template 1: Value Iteration (Algorithm 4.1)
```python
def value_iteration(env, gamma=0.9, theta=1e-6, max_iter=1000):
    """Value iteration: v_{k+1} = max_a [r(s,a) + gamma * sum p(s'|s,a) * v_k(s')]"""
    V = np.zeros(env.n_states)
    P = env.get_transition_model()

    for k in range(max_iter):
        V_new = np.zeros(env.n_states)
        for s in range(env.n_states):
            q_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for s_next in range(env.n_states):
                    if P[s, a, s_next] > 0:
                        r = env.rewards.get((s, a, s_next), -1)
                        q_values[a] += P[s, a, s_next] * (r + gamma * V[s_next])
            V_new[s] = np.max(q_values)

        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new

    # Extract greedy policy
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        q_values = np.zeros(env.n_actions)
        for a in range(env.n_actions):
            for s_next in range(env.n_states):
                if P[s, a, s_next] > 0:
                    r = env.rewards.get((s, a, s_next), -1)
                    q_values[a] += P[s, a, s_next] * (r + gamma * V[s_next])
        policy[s] = np.argmax(q_values)

    return V, policy
```

#### Template 2: Policy Iteration (Algorithm 4.2)
```python
def policy_iteration(env, gamma=0.9, theta=1e-6, max_iter=1000):
    """Policy iteration: evaluate pi exactly, then improve greedily."""
    P = env.get_transition_model()
    policy = np.zeros(env.n_states, dtype=int)  # initial policy (all action 0)
    V = np.zeros(env.n_states)

    for iteration in range(max_iter):
        # Policy evaluation: solve v_pi = r_pi + gamma * P_pi * v_pi iteratively
        for _ in range(max_iter):
            V_new = np.zeros(env.n_states)
            for s in range(env.n_states):
                a = policy[s]
                for s_next in range(env.n_states):
                    if P[s, a, s_next] > 0:
                        r = env.rewards.get((s, a, s_next), -1)
                        V_new[s] += P[s, a, s_next] * (r + gamma * V[s_next])
            if np.max(np.abs(V_new - V)) < theta:
                break
            V = V_new

        # Policy improvement: pi'(s) = argmax_a q_pi(s,a)
        policy_stable = True
        for s in range(env.n_states):
            q_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for s_next in range(env.n_states):
                    if P[s, a, s_next] > 0:
                        r = env.rewards.get((s, a, s_next), -1)
                        q_values[a] += P[s, a, s_next] * (r + gamma * V[s_next])
            new_action = np.argmax(q_values)
            if new_action != policy[s]:
                policy_stable = False
            policy[s] = new_action

        if policy_stable:
            break

    return V, policy
```

#### Template 3: MC Epsilon-Greedy (Algorithm 5.3)
```python
def mc_epsilon_greedy(env, n_episodes=5000, gamma=0.9, epsilon=0.1):
    """MC exploring starts with epsilon-greedy: uses complete episode returns."""
    Q = np.zeros((env.n_states, env.n_actions))
    returns_count = np.zeros((env.n_states, env.n_actions))

    for episode in range(n_episodes):
        # Generate episode using epsilon-greedy policy
        trajectory = []
        state = np.random.randint(env.n_states)

        for step in range(200):
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])
            next_state, reward = env.step(state, action)
            trajectory.append((state, action, reward))
            state = next_state
            if state == env.target_state:
                break

        # Backward return computation (first-visit MC)
        G = 0
        visited = set()
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            G = r_t + gamma * G
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                returns_count[s_t, a_t] += 1
                # Incremental mean update
                Q[s_t, a_t] += (G - Q[s_t, a_t]) / returns_count[s_t, a_t]

    policy = np.argmax(Q, axis=1)
    return Q, policy
```

#### Template 4: Sarsa (Algorithm 7.1)
```python
def sarsa(env, n_episodes=5000, gamma=0.9, alpha_0=0.5, epsilon_0=0.1):
    """Sarsa: q(s,a) += alpha * [r + gamma * q(s',a') - q(s,a)] (on-policy)"""
    Q = np.zeros((env.n_states, env.n_actions))
    visit_count = np.zeros((env.n_states, env.n_actions))

    for episode in range(n_episodes):
        state = np.random.randint(env.n_states)
        epsilon = epsilon_0 / (1 + episode / 1000)

        # Choose initial action (epsilon-greedy)
        if np.random.random() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state])

        for step in range(200):
            next_state, reward = env.step(state, action)
            visit_count[state, action] += 1

            # Choose next action from SAME policy (on-policy)
            if np.random.random() < epsilon:
                next_action = np.random.randint(env.n_actions)
            else:
                next_action = np.argmax(Q[next_state])

            alpha = alpha_0 / visit_count[state, action]

            # Sarsa update: uses q(s',a') not max_a' q(s',a')
            td_target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            action = next_action
            if state == env.target_state:
                break

    policy = np.argmax(Q, axis=1)
    return Q, policy
```

#### Template 5: Q-Learning (Algorithm 7.x)
```python
def q_learning(env, n_episodes=5000, gamma=0.9, alpha_0=0.5, epsilon_0=0.1):
    """Q-learning: q(s,a) += alpha * [r + gamma * max_a' q(s',a') - q(s,a)]"""
    Q = np.zeros((env.n_states, env.n_actions))
    visit_count = np.zeros((env.n_states, env.n_actions))

    for episode in range(n_episodes):
        state = np.random.randint(env.n_states)  # random start

        for step in range(200):  # max steps per episode
            # Epsilon-greedy action selection
            epsilon = epsilon_0 / (1 + episode / 1000)
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[state])

            next_state, reward = env.step(state, action)
            visit_count[state, action] += 1

            # Learning rate: alpha = 1/n(s,a) satisfies SA conditions
            alpha = alpha_0 / visit_count[state, action]

            # Q-learning update (off-policy: uses max)
            td_target = reward + gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            if state == env.target_state:
                break

    policy = np.argmax(Q, axis=1)
    return Q, policy
```

#### Template 6: REINFORCE (Algorithm 9.1)
```python
def reinforce(env, n_episodes=5000, gamma=0.99, alpha=0.01):
    """REINFORCE: theta += alpha * grad ln pi(a|s,theta) * G_t"""
    # Softmax policy: pi(a|s) = exp(theta[s,a]) / sum_a' exp(theta[s,a'])
    theta = np.zeros((env.n_states, env.n_actions))

    def softmax_policy(state):
        logits = theta[state] - np.max(theta[state])  # numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    for episode in range(n_episodes):
        # Generate episode
        trajectory = []
        state = np.random.randint(env.n_states)

        for step in range(200):
            probs = softmax_policy(state)
            action = np.random.choice(env.n_actions, p=probs)
            next_state, reward = env.step(state, action)
            trajectory.append((state, action, reward))
            state = next_state
            if state == env.target_state:
                break

        # Compute returns and update
        G = 0
        for t in reversed(range(len(trajectory))):
            s_t, a_t, r_t = trajectory[t]
            # G_t is an unbiased sample estimate of q_pi(s_t, a_t) — see KB Alg 9.1
            G = r_t + gamma * G  # G_t = r_{t+1} + gamma * G_{t+1}

            # Grad ln pi for softmax: e_a - pi(.|s)
            probs = softmax_policy(s_t)
            grad_ln_pi = -probs.copy()
            grad_ln_pi[a_t] += 1.0

            # REINFORCE update
            theta[s_t] += alpha * (gamma ** t) * G * grad_ln_pi

    return theta, softmax_policy
```

#### Template 7: A2C — Advantage Actor-Critic (Algorithm 10.2)
```python
def a2c(env, n_episodes=5000, gamma=0.99, alpha_theta=0.01, alpha_w=0.05):
    """A2C: actor (theta) + critic (w) with TD error as advantage estimate."""
    # Separate actor and critic parameters
    theta = np.zeros((env.n_states, env.n_actions))  # actor (policy)
    w = np.zeros(env.n_states)                        # critic (state value)

    def softmax_policy(state):
        logits = theta[state] - np.max(theta[state])
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    for episode in range(n_episodes):
        state = np.random.randint(env.n_states)

        for step in range(200):
            probs = softmax_policy(state)
            action = np.random.choice(env.n_actions, p=probs)
            next_state, reward = env.step(state, action)

            # TD error = advantage estimate: delta = r + gamma * V(s') - V(s)
            delta = reward + gamma * w[next_state] - w[state]

            # Critic update: w += alpha_w * delta * grad_w V(s)
            w[state] += alpha_w * delta

            # Actor update: theta += alpha_theta * delta * grad ln pi(a|s)
            grad_ln_pi = -probs.copy()
            grad_ln_pi[action] += 1.0
            theta[state] += alpha_theta * delta * grad_ln_pi

            state = next_state
            if state == env.target_state:
                break

    return theta, w, softmax_policy
```

#### Template 8: TD(0) with Linear Function Approximation (Ch 8)
```python
def td_linear(env, feature_fn, n_episodes=5000, gamma=0.9, alpha_0=0.01):
    """TD(0) with linear FA: w += alpha * (r + gamma * phi(s')^T w - phi(s)^T w) * phi(s)"""
    d = feature_fn(0).shape[0]  # feature dimension
    w = np.zeros(d)

    for episode in range(n_episodes):
        state = np.random.randint(env.n_states)
        alpha = alpha_0 / (1 + episode / 1000)

        for step in range(200):
            # Follow some behavior policy (e.g., uniform random for exploration)
            action = np.random.randint(env.n_actions)
            next_state, reward = env.step(state, action)

            phi_s = feature_fn(state)
            phi_s_next = feature_fn(next_state)

            # TD(0) update with linear FA
            td_error = reward + gamma * phi_s_next @ w - phi_s @ w
            w += alpha * td_error * phi_s

            state = next_state
            if state == env.target_state:
                break

    return w
```

#### Template 9: DQN (Algorithm 8.3 extended)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_dqn(env, state_dim, action_dim, n_episodes=10000, gamma=0.99,
              lr=1e-3, buffer_size=10000, batch_size=64, target_update=100):
    """DQN: J = E[(r + gamma * max_a' q(s',a'; w_T) - q(s,a; w))^2]"""
    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    replay_buffer = deque(maxlen=buffer_size)

    for episode in range(n_episodes):
        state = env.reset()
        epsilon = max(0.01, 1.0 - episode / (n_episodes * 0.5))

        for step in range(200):
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(state))
                    action = q_vals.argmax().item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            # Train from replay buffer
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_t = torch.FloatTensor(np.array(states))
                actions_t = torch.LongTensor(actions)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.FloatTensor(np.array(next_states))
                dones_t = torch.FloatTensor(dones)

                # Current Q values
                q_values = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

                # Target Q values (from target network)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(1)[0]
                    targets = rewards_t + gamma * next_q * (1 - dones_t)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

    return q_net
```

## Learning Rate Schedules

### SA-Compliant Schedules (sum=inf, sum-sq<inf)
```python
# Schedule 1: Harmonic (1/k)
alpha_k = 1.0 / (visit_count + 1)

# Schedule 2: Polynomial decay
alpha_k = alpha_0 / (1 + k / tau)  # tau controls decay speed

# Schedule 3: Per-state-action (recommended for tabular)
alpha_k_sa = 1.0 / n_visits[s, a]
```

### Practical Schedules (constant or slow decay)
```python
# Constant (does NOT satisfy SA conditions, but works with target networks)
alpha = 1e-3  # for DQN, DDPG

# Linear decay
alpha_k = alpha_0 * (1 - k / n_total)

# Exponential decay
alpha_k = alpha_0 * decay_rate ** k
```

## Feature Engineering

### Tabular (one-hot) Features
```python
def tabular_features(state, n_states):
    phi = np.zeros(n_states)
    phi[state] = 1.0
    return phi
```

### Polynomial Features
```python
def polynomial_features(state, degree=3, size=3):
    r, c = state // size, state % size
    # Normalize to [0, 1]
    x, y = r / (size - 1), c / (size - 1)
    features = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            features.append(x**i * y**j)
    return np.array(features)
```

### Fourier Basis Features
```python
def fourier_features(state, order=3, size=3):
    r, c = state // size, state % size
    x, y = r / (size - 1), c / (size - 1)
    features = []
    for i in range(order + 1):
        for j in range(order + 1):
            features.append(np.cos(np.pi * (i * x + j * y)))
    return np.array(features)
```

## Implementation Checklist
- [ ] Environment returns correct transitions and rewards
- [ ] Learning rate satisfies SA conditions (for convergence guarantees)
- [ ] Epsilon-greedy ensures sufficient exploration
- [ ] Q-values initialized (zeros or optimistic)
- [ ] Episode termination handled correctly
- [ ] Return computation uses correct discount factor
- [ ] For FA: features normalized, gradient computed correctly
- [ ] For DQN: target network updated at correct frequency
- [ ] For DQN: replay buffer large enough, batch size appropriate
- [ ] For policy gradient: log-probability gradient correct for chosen parameterization

## Debugging Guide
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Values diverge | Learning rate too high | Reduce alpha or use SA schedule |
| No learning | Learning rate too low or no exploration | Increase alpha_0 or epsilon |
| Oscillating values | Constant alpha with noise | Use decreasing alpha_k |
| Wrong policy | Reward function incorrect | Verify r(s,a,s') |
| Slow convergence | Poor exploration | Increase epsilon, use exploring starts |
| FA divergence | Deadly triad | Use target networks, reduce bootstrap |
