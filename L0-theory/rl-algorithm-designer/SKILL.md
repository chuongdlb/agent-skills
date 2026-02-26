---
name: rl-algorithm-designer
description: >
  Designs new RL algorithms by composing known design patterns, generating update rules from first principles, and ensuring mathematical soundness.
layer: L0
domain: [general-rl]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: [rl-theory-analyzer, rl-convergence-prover]
tags: [algorithm-design, patterns, theory]
---

# RL Algorithm Designer

## Purpose
Design new RL algorithms by composing known design patterns, generating update rules from first principles, and ensuring mathematical soundness.

## When to Use
Invoke this skill when you need to:
- Design a new RL algorithm for a specific problem setting
- Combine multiple design patterns into a coherent algorithm
- Generate update rules from a specified objective function
- Propose convergence-aware modifications to existing algorithms
- Adapt an existing algorithm to a new setting

## Design Patterns Library

### Pattern 1: Generalized Policy Iteration (GPI)
**What:** Alternating between policy evaluation and policy improvement
**When:** Any iterative RL algorithm
**How:**
- Evaluation: Estimate v_pi or q_pi for current policy
- Improvement: Derive better policy from current value estimates
**Variants:** Full evaluation (PI), single-step evaluation (VI), truncated (TPI with j steps)

### Pattern 2: Bootstrapping
**What:** Using current value estimates as part of the target
**When:** Want to learn online without waiting for episode completion
**How:** Replace full return G_t with r_{t+1} + gamma * v(s_{t+1})
**Trade-off:** Introduces bias but reduces variance; enables online learning
**Risk:** Part of the deadly triad with FA + off-policy

### Pattern 3: Stochastic Approximation
**What:** Replacing expectations with single samples
**When:** Model-free setting where E[...] cannot be computed exactly
**How:**
- Replace E_pi[...] with single sample from behavior
- Use diminishing step sizes: sum(alpha_k)=inf, sum(alpha_k^2)<inf
**Foundation:** Robbins-Monro theorem guarantees convergence under conditions

### Pattern 4: Epsilon-Greedy Exploration
**What:** Choose greedy action with prob 1-eps, random with prob eps
**When:** Need exploration in discrete action spaces
**How:** pi(a|s) = 1-eps+eps/|A| for a=a*, eps/|A| otherwise
**Variants:** Decaying epsilon, Boltzmann exploration, UCB

### Pattern 5: Experience Replay
**What:** Store transitions in buffer, sample uniformly for training
**When:** Off-policy algorithms with function approximation
**How:** Buffer D = {(s,a,r,s')}, sample mini-batch for each update
**Benefits:** Breaks correlation between consecutive samples, reuses data efficiently
**Used in:** DQN, DDPG, SAC, TD3

### Pattern 6: Target Networks
**What:** Separate network for computing targets, updated slowly
**When:** Using FA with bootstrapping (to stabilize targets)
**How:** w_target updated via w_T ← w every C steps, or w_T ← tau*w + (1-tau)*w_T
**Benefits:** Prevents moving target problem, stabilizes training
**Used in:** DQN, DDPG, TD3

### Pattern 7: Importance Sampling
**What:** Correcting for distribution mismatch between behavior and target policy
**When:** Off-policy learning
**How:** Weight updates by rho = pi(a|s) / beta(a|s)
**Variants:** Ordinary IS, weighted IS, per-decision IS
**Risk:** High variance when policies diverge significantly

### Pattern 8: Baseline Subtraction
**What:** Subtracting a state-dependent baseline from returns
**When:** Policy gradient methods (to reduce variance)
**How:** Replace q_pi(s,a) with q_pi(s,a) - b(s) in gradient estimate
**Key property:** E[grad ln pi * b(S)] = 0 (unbiased)
**Best baseline:** b(s) = v_pi(s), giving advantage A(s,a) = q_pi(s,a) - v_pi(s)

### Pattern 9: Function Approximation
**What:** Parameterize value/policy with function of features
**When:** Large or continuous state/action spaces
**How:** v_hat(s,w) = phi(s)^T * w (linear) or neural network (nonlinear)
**Benefits:** Generalization across similar states
**Risk:** Approximation error, potential divergence with bootstrapping + off-policy

### Pattern 10: Actor-Critic Decomposition
**What:** Separate networks for policy (actor) and value (critic)
**When:** Want benefits of both policy gradient and value-based methods
**How:**
- Critic: Learn q_pi or v_pi using TD methods
- Actor: Update policy using policy gradient with critic's estimates
**Benefits:** Lower variance than pure PG, handles continuous actions

## Algorithm Design Procedure

### Step 1: Define the Problem Setting
- State space: discrete or continuous?
- Action space: discrete or continuous?
- Model availability: known or unknown?
- Episode structure: episodic or continuing?
- Data regime: online or batch?

### Step 2: Choose the Core Approach
Based on the setting:
| Setting | Recommended |
|---------|------------|
| Discrete S, A, known model | Dynamic Programming (VI/PI) |
| Discrete S, A, unknown model, episodic | MC or TD methods |
| Large/continuous S, discrete A | DQN-style (value + FA) |
| Continuous S and A | Actor-Critic (A2C, DDPG, SAC) |
| Need sample efficiency | Off-policy + replay buffer |

### Step 3: Select Design Patterns
Compose patterns based on needs:
- **Stability:** Add target networks (Pattern 6)
- **Exploration:** Add epsilon-greedy (Pattern 4) or entropy bonus
- **Variance reduction:** Add baseline (Pattern 8)
- **Sample efficiency:** Add experience replay (Pattern 5)
- **Off-policy correction:** Add importance sampling (Pattern 7)

### Step 4: Derive the Update Rule
From the objective J(theta) or J(w):

**For value-based:**
```
w_{t+1} = w_t + alpha * [target - v_hat(s_t, w_t)] * grad_w v_hat(s_t, w_t)
```
where target depends on the algorithm choice.

**For policy gradient:**
```
theta_{t+1} = theta_t + alpha * grad_theta ln pi(a_t|s_t, theta) * [G_t - b(s_t)]
```

**For actor-critic:**
```
Critic: w_{t+1} = w_t + alpha_w * delta_t * grad_w v_hat(s_t, w_t)
Actor: theta_{t+1} = theta_t + alpha_theta * delta_t * grad_theta ln pi(a_t|s_t, theta)
```
where delta_t = r_{t+1} + gamma * v_hat(s_{t+1}, w) - v_hat(s_t, w)

### Step 5: Verify Mathematical Soundness
- Check convergence conditions (reference kb-stochastic-approximation.md)
- Verify no deadly triad combination
- Ensure learning rate schedule satisfies SA conditions
- Check that the update targets the correct fixed point

### Step 6: Specify Hyperparameters
- Learning rate schedule: alpha_k = 1/k or alpha_k = alpha_0 / (1 + k/tau)
- Discount factor: gamma in [0.9, 0.999] typical
- Exploration: epsilon schedule, entropy coefficient
- Target network: update frequency C or soft update tau
- Replay buffer: size, mini-batch size, prioritization

## Innovation Dimensions
When designing novel algorithms, consider modifying along these axes:
1. **New objectives:** Different J(theta) beyond standard value/reward metrics
2. **New update rules:** Modified gradient estimators, multi-step targets
3. **New approximators:** Architecture choices (attention, graph networks, etc.)
4. **New exploration:** Curiosity-driven, count-based, posterior sampling
5. **New combination:** Mix patterns in novel ways (e.g., n-step + off-policy + replay)

## Output Format
When designing an algorithm, produce:
1. **Problem Setting:** (state/action spaces, model availability, etc.)
2. **Algorithm Name:** (descriptive name)
3. **Design Patterns Used:** (list of patterns composed)
4. **Update Rules:** (complete mathematical specification)
5. **Pseudocode:** (step-by-step algorithm)
6. **Hyperparameters:** (with recommended defaults)
7. **Convergence Notes:** (conditions and guarantees)
8. **Implementation Notes:** (practical considerations)
