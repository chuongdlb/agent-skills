---
chapter: 1
title: Basic Concepts
key_topics: [state, action, state transition, policy, reward, trajectory, return, discounted return, discount rate, episode, episodic task, continuing task, absorbing state, MDP, Markov property, Markov chain]
depends_on: []
required_by: [2, 3, 4, 5, 6, 7, 8, 9, 10]
---

# Chapter 1: Basic Concepts

> Source: *Mathematical Foundations of Reinforcement Learning* (S. Zhao, Springer 2025), Chapter 1, pp. 1-13
> Supplemented by: Lecture slides L1 (28 slides)
> Errata: No corrections for this chapter

## Purpose and Context

This chapter introduces the fundamental concepts of reinforcement learning through a **grid world example** that is reused throughout the entire book. The concepts are first demonstrated intuitively, then formalized under the **Markov Decision Process (MDP)** framework.

**Position in the book**: Chapter 1 is the foundation for all subsequent chapters. The book is structured in two parts:
- **Part 1 (Fundamental tools)**: Chapters 1-6 (basic concepts, Bellman equation, Bellman optimality equation, value/policy iteration, Monte Carlo methods, stochastic approximation)
- **Part 2 (Algorithms/Methods)**: Chapters 7-10 (TD methods, value function methods, policy gradient methods, actor-critic methods)

Key transitions across the book: *with model* -> *without model*; *tabular representation* -> *function representation*; *value-based* -> *policy-based* -> *combined (actor-critic)*.

---

## 1.1 The Grid World Example

The running example throughout the book is a **3x3 grid world** where a robot (agent) navigates cells.

**Setup:**
- White cells: accessible
- Orange cells: forbidden (accessible but penalized)
- One target cell the agent aims to reach
- Agent occupies exactly one cell per time step and can move to adjacent cells

**Core task**: Find a "good" policy that enables the agent to reach the target from any initial cell, where "good" means avoiding forbidden cells, unnecessary detours, and boundary collisions.

**Key distinction**: The task is trivial if the agent knows the map in advance. It becomes the RL problem when the agent has **no prior information** about the environment and must learn by **trial and error** through interaction.

---

## 1.2 State and Action

### State
- **Definition**: A state describes the agent's status with respect to the environment.
- In the grid world: state = agent's location. Nine cells -> nine states: $s_1, s_2, \ldots, s_9$.
- **State space**: $\mathcal{S} = \{s_1, \ldots, s_9\}$ — the set of all states.

### Action
- Five possible actions per state:
  - $a_1$: move upward
  - $a_2$: move rightward
  - $a_3$: move downward
  - $a_4$: move leftward
  - $a_5$: stay still
- **Action space**: $\mathcal{A} = \{a_1, \ldots, a_5\}$ — the set of all actions.
- Different states *can* have different action spaces $\mathcal{A}(s_i)$, e.g., $\mathcal{A}(s_1) = \{a_2, a_3, a_5\}$ if boundary actions are excluded.
- **Convention in this book**: The most general case is used — $\mathcal{A}(s_i) = \mathcal{A} = \{a_1, \ldots, a_5\}$ for all $i$.

---

## 1.3 State Transition

**Definition**: The process of moving from one state to another after taking an action.

**Notation**: $s_1 \xrightarrow{a_2} s_2$ means "at state $s_1$, taking action $a_2$, transition to $s_2$."

### Boundary Behavior
When the agent attempts to exit the grid boundary, it is **bounced back** to the same state:
$$s_1 \xrightarrow{a_1} s_1$$

### Forbidden Cell Behavior (Two scenarios)
1. **Scenario 1 (used in this book)**: Forbidden cells are **accessible** but penalized. $s_5 \xrightarrow{a_2} s_6$
2. **Scenario 2**: Forbidden cells are inaccessible (walls). Agent bounced back: $s_5 \xrightarrow{a_2} s_5$

The book uses Scenario 1 as it is more general and interesting.

### Complete State Transition Table

| | $a_1$ (up) | $a_2$ (right) | $a_3$ (down) | $a_4$ (left) | $a_5$ (stay) |
|---|---|---|---|---|---|
| $s_1$ | $s_1$ | $s_2$ | $s_4$ | $s_1$ | $s_1$ |
| $s_2$ | $s_2$ | $s_3$ | $s_5$ | $s_1$ | $s_2$ |
| $s_3$ | $s_3$ | $s_3$ | $s_6$ | $s_2$ | $s_3$ |
| $s_4$ | $s_1$ | $s_5$ | $s_7$ | $s_4$ | $s_4$ |
| $s_5$ | $s_2$ | $s_6$ | $s_8$ | $s_4$ | $s_5$ |
| $s_6$ | $s_3$ | $s_6$ | $s_9$ | $s_5$ | $s_6$ |
| $s_7$ | $s_4$ | $s_8$ | $s_7$ | $s_7$ | $s_7$ |
| $s_8$ | $s_5$ | $s_9$ | $s_8$ | $s_7$ | $s_8$ |
| $s_9$ | $s_6$ | $s_9$ | $s_9$ | $s_8$ | $s_9$ |

### Mathematical Formalization: State Transition Probability

State transitions are described by conditional probabilities $p(s'|s, a)$:

**Example** (deterministic): At $s_1$, taking $a_2$:
$$p(s_2|s_1, a_2) = 1, \quad p(s_i|s_1, a_2) = 0 \;\;\forall\; i \neq 2$$

**Stochastic transitions**: Possible when random factors exist (e.g., wind gusts). In such cases, $p(s_5|s_1, a_2) > 0$ even though $a_2$ nominally moves right.

**Convention in this book**: Only **deterministic** state transitions are used in grid world examples for simplicity.

**Key insight**: The tabular representation can only describe deterministic transitions. Conditional probability distributions are needed for the general (stochastic) case.

---

## 1.4 Policy

**Definition**: A policy tells the agent which actions to take at every state.

### Representations

1. **Intuitive**: Arrows on the grid indicating direction of movement.
2. **Mathematical**: Conditional probability $\pi(a|s)$ — the probability of choosing action $a$ in state $s$.

### Deterministic Policy Example
At $s_1$: $\pi(a_1|s_1) = 0,\; \pi(a_2|s_1) = 1,\; \pi(a_3|s_1) = 0,\; \pi(a_4|s_1) = 0,\; \pi(a_5|s_1) = 0$

This means the agent always moves right at $s_1$.

### Stochastic Policy Example
At $s_1$: $\pi(a_2|s_1) = 0.5,\; \pi(a_3|s_1) = 0.5$ (all others zero)

The agent moves right or down with equal probability.

### Tabular Representation of a Stochastic Policy

| | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |
|---|---|---|---|---|---|
| $s_1$ | 0 | 0.5 | 0.5 | 0 | 0 |
| $s_2$ | 0 | 0 | 1 | 0 | 0 |
| $s_3$ | 0 | 0 | 0 | 1 | 0 |
| $s_4$ | 0 | 1 | 0 | 0 | 0 |
| $s_5$ | 0 | 0 | 1 | 0 | 0 |
| $s_6$ | 0 | 0 | 1 | 0 | 0 |
| $s_7$ | 0 | 1 | 0 | 0 | 0 |
| $s_8$ | 0 | 1 | 0 | 0 | 0 |
| $s_9$ | 0 | 0 | 0 | 0 | 1 |

Each row sums to 1 (valid probability distribution). Tabular representation can express both deterministic and stochastic policies.

**Forward reference**: Chapter 8 introduces an alternative — policies as **parameterized functions** (function approximation).

---

## 1.5 Reward

**Definition**: A real-valued number $r$ obtained after executing an action at a state. Written as $r(s, a)$.

### Reward Design for the Grid World
| Condition | Reward |
|---|---|
| Agent attempts to exit boundary | $r_{\text{boundary}} = -1$ |
| Agent enters a forbidden cell | $r_{\text{forbidden}} = -1$ |
| Agent reaches target state | $r_{\text{target}} = +1$ |
| Otherwise | $r_{\text{other}} = 0$ |

### Complete Reward Table

| | $a_1$ (up) | $a_2$ (right) | $a_3$ (down) | $a_4$ (left) | $a_5$ (stay) |
|---|---|---|---|---|---|
| $s_1$ | $r_{\text{bound}}$ | 0 | 0 | $r_{\text{bound}}$ | 0 |
| $s_2$ | $r_{\text{bound}}$ | 0 | 0 | 0 | 0 |
| $s_3$ | $r_{\text{bound}}$ | $r_{\text{bound}}$ | $r_{\text{forbid}}$ | 0 | 0 |
| $s_4$ | 0 | 0 | $r_{\text{forbid}}$ | $r_{\text{bound}}$ | 0 |
| $s_5$ | 0 | $r_{\text{forbid}}$ | 0 | 0 | 0 |
| $s_6$ | 0 | $r_{\text{bound}}$ | $r_{\text{target}}$ | 0 | $r_{\text{forbid}}$ |
| $s_7$ | 0 | 0 | $r_{\text{bound}}$ | $r_{\text{bound}}$ | $r_{\text{forbid}}$ |
| $s_8$ | 0 | $r_{\text{target}}$ | $r_{\text{bound}}$ | $r_{\text{forbid}}$ | 0 |
| $s_9$ | $r_{\text{forbid}}$ | $r_{\text{bound}}$ | $r_{\text{bound}}$ | 0 | $r_{\text{target}}$ |

### Key Properties of Reward

1. **Positive reward** = encouragement; **negative reward** = discouragement.
2. **Target state $s_9$ behavior**: The process does not terminate at $s_9$. Taking $a_5$ at $s_9$ yields $r_{\text{target}} = +1$, but taking $a_2$ at $s_9$ yields $r_{\text{boundary}} = -1$.
3. **Human-machine interface**: Reward is the mechanism through which we guide agent behavior.
4. **Immediate reward is not sufficient** for selecting good policies — the total long-run reward (return) must be considered. An action with the greatest immediate reward may not yield the greatest total reward.

### Mathematical Formalization: Reward Probability
Rewards are described by conditional probabilities $p(r|s, a)$:

$$p(r = -1|s_1, a_1) = 1, \quad p(r \neq -1|s_1, a_1) = 0$$

This is deterministic. In general, reward processes can be **stochastic** (e.g., studying hard yields positive but uncertain reward).

---

## 1.6 Trajectories, Returns, and Episodes

### Trajectory
A **trajectory** is a state-action-reward chain:
$$s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9$$

### Return
The **return** is the sum of all rewards collected along a trajectory:
$$\text{return} = 0 + 0 + 0 + 1 = 1$$

Also called **total reward** or **cumulative reward**.

### Using Returns to Evaluate Policies
- Policy 1 (avoids forbidden cells): trajectory $s_1 \to s_2 \to s_5 \to s_8 \to s_9$, return $= 1$
- Policy 2 (passes through forbidden cell): trajectory $s_1 \to s_4 \to s_7 \to s_8 \to s_9$, return $= 0 + (-1) + 0 + 1 = 0$
- **Conclusion**: Policy 1 is better (greater return). Mathematical evaluation matches intuition.

### Immediate vs. Future Rewards
A return consists of:
- **Immediate reward**: obtained right after taking an action at the initial state
- **Future rewards**: all rewards obtained after leaving the initial state

It is possible that immediate reward is negative while future rewards are positive. **Decisions should be based on returns (total reward), not immediate rewards**, to avoid short-sighted behavior.

### Discounted Return
For **infinitely long trajectories**, the naive return diverges:
$$\text{return} = 0 + 0 + 0 + 1 + 1 + 1 + \cdots = \infty$$

**Solution**: Introduce a **discount rate** $\gamma \in (0, 1)$:
$$\text{discounted return} = r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 r_4 + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{k+1}$$

**Example**: For the trajectory $s_1 \to s_2 \to s_5 \to s_8 \to s_9 \to s_9 \to \cdots$:
$$\text{discounted return} = 0 + \gamma \cdot 0 + \gamma^2 \cdot 0 + \gamma^3 \cdot 1 + \gamma^4 \cdot 1 + \gamma^5 \cdot 1 + \cdots = \gamma^3 \cdot \frac{1}{1-\gamma}$$

### Roles of the Discount Rate $\gamma$
1. **Makes infinite sums finite**: Ensures convergence of the return.
2. **Balances near vs. far future rewards**:
   - $\gamma \to 0$: Agent is **short-sighted**, emphasizes near-future rewards.
   - $\gamma \to 1$: Agent is **far-sighted**, willing to accept short-term negative rewards for long-term gain.

**Forward reference**: The effects of $\gamma$ are demonstrated in detail in Section 3.5.

### Episodes
- **Episode** (or trial): A trajectory that terminates at a **terminal state**. Usually assumed to be finite.
- **Episodic tasks**: Tasks with episodes (finite trajectories ending at terminal states).
- **Continuing tasks**: Tasks with no terminal states; interaction never ends.

### Unifying Episodic and Continuing Tasks
Episodic tasks can be converted to continuing tasks by defining what happens after reaching the terminal state:

1. **Absorbing state approach**: Design the terminal state so the agent stays forever.
   - Option A: Set $\mathcal{A}(s_9) = \{a_5\}$ (only "stay" action).
   - Option B: Set $p(s_9|s_9, a_i) = 1$ for all $i$ (all actions lead back to $s_9$).
2. **Normal state approach (used in this book)**: Treat the terminal state as a normal state with $\mathcal{A}(s_9) = \{a_1, \ldots, a_5\}$. The agent can leave and return. Since $r_{\text{target}} = +1$ is obtained each time $s_9$ is reached, the agent eventually learns to stay. A **discount rate is required** to prevent divergence.

---

## 1.7 Markov Decision Processes (MDPs)

An MDP formalizes all the preceding concepts into a unified framework.

### Key Ingredients

#### Sets
| Component | Notation | Description |
|---|---|---|
| State space | $\mathcal{S}$ | Set of all states |
| Action space | $\mathcal{A}(s)$ | Set of actions available in state $s$ |
| Reward set | $\mathcal{R}(s,a)$ | Set of possible rewards for state-action pair $(s,a)$ |

#### Model (Dynamics)
| Component | Notation | Constraint |
|---|---|---|
| State transition probability | $p(s'|s,a)$ | $\sum_{s' \in \mathcal{S}} p(s'|s,a) = 1$ for any $(s,a)$ |
| Reward probability | $p(r|s,a)$ | $\sum_{r \in \mathcal{R}(s,a)} p(r|s,a) = 1$ for any $(s,a)$ |

#### Policy
$$\pi(a|s): \quad \sum_{a \in \mathcal{A}(s)} \pi(a|s) = 1 \text{ for any } s \in \mathcal{S}$$

#### Markov Property (Memoryless Property)
$$p(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = p(s_{t+1}|s_t, a_t)$$
$$p(r_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = p(r_{t+1}|s_t, a_t)$$

**Meaning**: The next state and reward depend **only** on the current state and action, not on history. This property is critical for deriving the **Bellman equation** (Chapter 2).

### Model Properties
- **Stationary (time-invariant)**: Model does not change over time. *Used in this book.*
- **Non-stationary (time-variant)**: Model may change (e.g., forbidden areas appearing/disappearing).

### MDP vs. Markov Process (MP)
- Once the policy $\pi$ in an MDP is **fixed**, the MDP degenerates into a **Markov Process** (MP).
- In discrete-time with finite/countable states, an MP is also called a **Markov Chain**.
- This book uses "Markov process" and "Markov chain" interchangeably.
- This book mainly considers **finite MDPs** (finite states and actions) — the simplest case that should be fully understood first.

### Agent-Environment Interaction Loop
The RL framework is a closed loop:
1. Agent senses its **state**
2. Agent selects an **action** based on its **policy**
3. Actuator executes the action
4. Environment transitions to a new state and emits a **reward**
5. Agent interprets the new state and reward via interpreters
6. Repeat

---

## 1.9 Q&A — Important Clarifications

### Q: Can all rewards be set as negative (or all positive)?
**A**: Yes. It is the **relative** reward values, not absolute values, that determine the optimal policy. Adding a constant to all rewards does not change the optimal policy:
- Original: $r_{\text{boundary}} = -1, r_{\text{forbidden}} = -1, r_{\text{target}} = +1, r_{\text{other}} = 0$
- Shifted by $-2$: $r_{\text{boundary}} = -3, r_{\text{forbidden}} = -3, r_{\text{target}} = -1, r_{\text{other}} = -2$
- **Same optimal policy** because optimal policies are invariant to **affine transformations** of rewards.

**Forward reference**: This is proven formally in Chapter 3, Section 3.5.

### Q: Should the reward depend on the next state $s'$?
**A**: The reward $r$ does depend on $s$, $a$, and $s'$. However, since $s'$ itself depends on $s$ and $a$, we can write $r$ as a function of $(s, a)$ alone:
$$p(r|s,a) = \sum_{s'} p(r|s,a,s') \cdot p(s'|s,a)$$

This simplification allows the **Bellman equation** to be established cleanly (Chapter 2).

---

## Concept Index

| Concept | Notation | Section |
|---|---|---|
| State | $s \in \mathcal{S}$ | 1.2 |
| Action | $a \in \mathcal{A}(s)$ | 1.2 |
| State space | $\mathcal{S}$ | 1.2 |
| Action space | $\mathcal{A}(s)$ or $\mathcal{A}$ | 1.2 |
| State transition | $s \xrightarrow{a} s'$ | 1.3 |
| State transition probability | $p(s'|s,a)$ | 1.3 |
| Policy | $\pi(a|s)$ | 1.4 |
| Deterministic policy | $\pi(a|s) \in \{0, 1\}$ | 1.4 |
| Stochastic policy | $\pi(a|s) \in [0, 1]$ | 1.4 |
| Reward | $r(s,a)$ or $p(r|s,a)$ | 1.5 |
| Trajectory | state-action-reward chain | 1.6 |
| Return (total/cumulative reward) | $\sum r_k$ | 1.6 |
| Discounted return | $\sum \gamma^k r_{k+1}$ | 1.6 |
| Discount rate | $\gamma \in (0,1)$ | 1.6 |
| Episode / trial | finite trajectory to terminal state | 1.6 |
| Episodic task | task with terminal states | 1.6 |
| Continuing task | task without terminal states | 1.6 |
| Absorbing state | state the agent never leaves | 1.6 |
| Markov Decision Process (MDP) | $(\mathcal{S}, \mathcal{A}, p, r, \pi)$ | 1.7 |
| Markov property | $p(s_{t+1}|s_t,a_t)$ memoryless | 1.7 |
| Markov process / Markov chain | MDP with fixed policy | 1.7 |
| Model / dynamics | $p(s'|s,a)$ and $p(r|s,a)$ | 1.7 |
| Stationary vs. non-stationary model | time-invariant vs. time-variant | 1.7 |
| Finite MDP | finite $|\mathcal{S}|$ and $|\mathcal{A}|$ | 1.7 |

---

## Dependencies and Forward References

| This chapter concept | Used in / Extended by |
|---|---|
| State, action, policy, reward | Every subsequent chapter |
| State transition probability $p(s'|s,a)$ | Ch 2 (Bellman equation derivation) |
| Markov property | Ch 2 (enables Bellman equation) |
| Discount rate $\gamma$ | Ch 2, 3 (Bellman equations); Ch 3.5 (effect analysis) |
| Reward invariance to affine transforms | Ch 3.5 (formal proof) |
| Reward as function of $(s,a)$ vs $(s,a,s')$ | Ch 2 (Bellman equation formulation) |
| Tabular policy representation | Ch 8 (replaced by parameterized function approximation) |
| Finite MDP assumption | Chs 1-7 (tabular methods); relaxed in Chs 8-10 |
