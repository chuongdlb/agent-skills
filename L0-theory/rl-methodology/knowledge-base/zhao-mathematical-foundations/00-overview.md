---
type: overview
title: Knowledge Base Overview
---

# Knowledge Base Overview

This knowledge base distills the textbook **Mathematical Foundations of Reinforcement Learning** by Shiyu Zhao (Springer, 2025) into structured reference files suitable for AI-assisted study and retrieval. The book comprises **10 chapters** plus a **mathematical appendix**, spanning pages 1--270, organized in two parts:

- **Part 1 -- Fundamental Tools** (Chapters 1--6): Builds the conceptual and mathematical machinery (MDPs, Bellman equations, dynamic programming, Monte Carlo estimation, stochastic approximation).
- **Part 2 -- Algorithms / Methods** (Chapters 7--10): Presents the core RL algorithms (temporal-difference methods, value function approximation, policy gradient methods, actor-critic methods).

The appendix (Appendices A--D) provides prerequisite mathematics: probability theory, measure theory, convergence theorems, and gradient descent foundations.

---

## Reading Order and Dependency Diagram

```
                    +-----------------+
                    | Ch 1: Basic     |
                    | Concepts        |
                    +--------+--------+
                             |
                    +--------v--------+
                    | Ch 2: Bellman   |
                    | Equation        |
                    +--------+--------+
                             |
                    +--------v--------+
                    | Ch 3: Bellman   |
                    | Optimality Eq.  |
                    +--------+--------+
                             |
                    +--------v--------+
                    | Ch 4: Value &   |
                    | Policy Iteration|
                    +--------+--------+
                             |
                    +--------v--------+
                    | Ch 5: Monte     |
                    | Carlo Methods   |
                    +--------+--------+
                             |
                    +--------v--------+
                    | Ch 6: Stochastic|
                    | Approximation   |
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+                   |
     | Ch 7: Temporal-  |                   |
     | Difference       |                   |
     +--------+---------+                   |
              |                             |
     +--------v--------+                   |
     | Ch 8: Value      |                   |
     | Function Methods |                   |
     +--------+---------+                   |
              |                             |
     +--------v--------+          +--------v--------+
     | Ch 9: Policy     |<---------| Ch 6 (SGD)      |
     | Gradient Methods |          +-----------------+
     +--------+---------+
              |
     +--------v--------+
     | Ch 10: Actor-    |
     | Critic Methods   |
     +-----------------+

  Appendices A-D: Referenced throughout all chapters
```

**Dependency notes**: Each chapter depends on all preceding chapters. Chapter 9 also draws directly on the SGD foundations from Chapter 6 and on function approximation ideas from Chapter 8. Chapter 10 combines the value-based approach (Chapter 8) with the policy-based approach (Chapter 9).

---

## Chapter Summary Table

| Chapter | Title | Key Topics | Pages | Algorithms Covered |
|---------|-------|------------|-------|--------------------|
| 1 | Basic Concepts | State, action, policy, reward, trajectory, return, discount rate, MDP, Markov property | 1--13 | (none -- foundational definitions) |
| 2 | State Values and Bellman Equation | State value, action value, Bellman equation (elementwise & matrix-vector), policy evaluation, bootstrapping | 15--34 | Iterative policy evaluation, closed-form solution |
| 3 | Optimal State Values and Bellman Optimality Equation | Optimal policy, optimal state value, BOE, contraction mapping theorem, greedy policy, reward invariance | 35--55 | Iterative BOE solver (precursor to value iteration) |
| 4 | Value Iteration and Policy Iteration | Value iteration, policy iteration, truncated policy iteration, GPI | 57--76 | Value Iteration (Alg 4.1), Policy Iteration (Alg 4.2), Truncated Policy Iteration (Alg 4.3) |
| 5 | Monte Carlo Methods | Model-free transition, mean estimation, episode sampling, visit strategies, exploration vs exploitation | 77--99 | MC Basic (Alg 5.1), MC Exploring Starts (Alg 5.2), MC epsilon-Greedy (Alg 5.3) |
| 6 | Stochastic Approximation | Robbins-Monro algorithm, Dvoretzky's theorem, SGD, step size conditions, convergence theory | 101--124 | RM algorithm, SGD (batch, mini-batch, stochastic) |
| 7 | Temporal-Difference Methods | TD learning, TD error, TD target, Sarsa, Expected Sarsa, n-step Sarsa, Q-learning, on/off-policy | 125--150 | Sarsa (Alg 7.1), Q-learning (Algs 7.2, 7.3), Expected Sarsa, n-step Sarsa |
| 8 | Value Function Methods | Function approximation, linear approximation, stationary distribution, objective functions, DQN | 151--189 | TD-Linear, Sarsa with FA (Alg 8.2), Q-learning with FA (Alg 8.3), DQN, LSTD |
| 9 | Policy Gradient Methods | Parameterized policy, softmax policy, policy gradient theorem, metrics, log-derivative trick | 191--214 | REINFORCE (Alg 9.1) |
| 10 | Actor-Critic Methods | QAC, A2C, advantage function, baseline invariance, off-policy AC, importance sampling, DPG, DDPG | 215--236 | QAC (Alg 10.1), A2C (Alg 10.2), Off-policy AC (Alg 10.3), DPG (Alg 10.4) |
| App | Mathematical Appendix | Probability theory, measure theory, martingales, convergence theorems, gradient descent, convexity | 237--270 | (none -- mathematical reference) |

---

## Concept Flow Diagram

The book's intellectual progression can be summarized as follows:

```
MDP Fundamentals          Bellman Equation          Bellman Optimality Eq.
(Ch 1: states, actions,   (Ch 2: state values,      (Ch 3: optimal values,
 policies, rewards)        policy evaluation)         contraction mapping)
        |                         |                          |
        +----------+--------------+----------+---------------+
                   |                         |
           Value Iteration &           Monte Carlo
           Policy Iteration            Methods
           (Ch 4: DP algorithms)       (Ch 5: model-free,
                   |                    episode-based)
                   |                         |
                   +----------+--------------+
                              |
                   Stochastic Approximation
                   (Ch 6: RM algorithm, SGD,
                    convergence theory)
                              |
                   Temporal-Difference Methods
                   (Ch 7: TD, Sarsa, Q-learning,
                    on/off-policy, incremental)
                              |
                   Value Function Approximation
                   (Ch 8: linear approx, DQN,
                    tabular -> function)
                              |
                   Policy Gradient Methods
                   (Ch 9: REINFORCE, policy
                    gradient theorem)
                              |
                   Actor-Critic Methods
                   (Ch 10: QAC, A2C, DPG, DDPG,
                    value + policy combined)
```

**Key transitions across the book**:
1. **Model-based -> Model-free**: Chapters 2--4 require the model; Chapters 5, 7--10 learn from data.
2. **Tabular -> Function approximation**: Chapters 1--7 use tables; Chapters 8--10 use parameterized functions.
3. **Value-based -> Policy-based -> Actor-Critic**: Chapters 2--8 focus on values; Chapter 9 on policies; Chapter 10 combines both.
4. **On-policy -> Off-policy**: Sarsa is on-policy; Q-learning is off-policy; Chapter 10 develops off-policy actor-critic.

---

## File Listing

| File | Description |
|------|-------------|
| `00-overview.md` | This file. Knowledge base overview, reading order, and navigation guide. |
| `01-basic-concepts.md` | Chapter 1: MDP fundamentals -- states, actions, policies, rewards, returns, discount rate, Markov property. |
| `02-bellman-equation.md` | Chapter 2: State/action values, Bellman equation (elementwise and matrix-vector forms), policy evaluation, bootstrapping. |
| `03-bellman-optimality-equation.md` | Chapter 3: Optimal policies, Bellman optimality equation, contraction mapping theorem, greedy policy extraction. |
| `04-value-iteration-policy-iteration.md` | Chapter 4: Value iteration, policy iteration, truncated policy iteration, generalized policy iteration (GPI). |
| `05-monte-carlo-methods.md` | Chapter 5: MC Basic, MC Exploring Starts, MC epsilon-Greedy; model-free transition, visit strategies, exploration. |
| `06-stochastic-approximation.md` | Chapter 6: Robbins-Monro algorithm, Dvoretzky's theorem, SGD; mathematical bridge to incremental algorithms. |
| `07-temporal-difference-methods.md` | Chapter 7: TD learning, Sarsa, Expected Sarsa, n-step Sarsa, Q-learning; on-policy vs off-policy; unified TD viewpoint. |
| `08-value-function-methods.md` | Chapter 8: Function approximation (linear, polynomial, Fourier, neural network), DQN, experience replay, target networks. |
| `09-policy-gradient-methods.md` | Chapter 9: Parameterized policies, policy gradient theorem, REINFORCE, softmax policy, average value/reward metrics. |
| `10-actor-critic-methods.md` | Chapter 10: QAC, A2C, off-policy actor-critic, DPG, DDPG; importance sampling; baseline invariance. |
| `11-appendix.md` | Appendices A--D: Probability theory, measure theory, convergence theorems, gradient descent and convexity. |
| `12-cross-reference-index.md` | Cross-reference index: master concept index, algorithm comparison, theorem index, key equations, transition summaries. |

---

## Navigation Tips for AI Agents

1. **Start with this file** (`00-overview.md`) to understand the knowledge base structure and chapter dependencies before diving into specific topics.

2. **For concept lookups**, consult `12-cross-reference-index.md` first. It maps every major concept, algorithm, and theorem to the chapter(s) where it is defined and used.

3. **Follow the dependency chain**. Each chapter file has a YAML `depends_on` field listing prerequisite chapters. If a concept in Chapter 7 is unclear, trace back through Chapters 6, 5, 4, 3, 2, 1 as needed.

4. **Use the `required_by` YAML field** to understand forward impact. For instance, Chapter 2's Bellman equation is required by nearly every subsequent chapter.

5. **For algorithm questions**, the chapter files contain pseudocode, convergence guarantees, and detailed update rules. The cross-reference index (`12-cross-reference-index.md`) provides a quick comparison table of all algorithms.

6. **For mathematical prerequisites**, consult `11-appendix.md`. Key results include probability fundamentals (Appendix A), measure-theoretic foundations (Appendix B), convergence theorems used in Chapters 6--7 proofs (Appendix C), and gradient descent theory underlying Chapters 6, 8--10 (Appendix D).

7. **Grid world example**: The 3x3 grid world introduced in Chapter 1 is the running example throughout the book. Chapter 1 contains the canonical setup; subsequent chapters show how each algorithm applies to this same example.

8. **Chapter file structure**: Each chapter file follows a consistent format: YAML frontmatter, purpose/context section, section-by-section content with definitions, theorems, algorithms, worked examples, and a summary with key takeaways.
