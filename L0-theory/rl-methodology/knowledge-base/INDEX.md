# Knowledge Base Index

Master manifest of all knowledge base sources used by the `rl-methodology` skill.

## Sources

### zhao-mathematical-foundations
- **Full title:** Mathematical Foundations of Reinforcement Learning (Shiyu Zhao, 2024)
- **Directory:** `zhao-mathematical-foundations/`
- **Files:** 13
- **Size:** ~428 KB
- **Coverage:** MDP fundamentals, Bellman equations, value/policy iteration, Monte Carlo, stochastic approximation, TD methods, value function approximation, policy gradient, actor-critic

| File | Topic | Book Chapters |
|------|-------|---------------|
| `00-overview.md` | Book structure, notation, chapter summaries | All |
| `01-basic-concepts.md` | MDPs, states, actions, rewards, policies, returns | Ch 1 |
| `02-bellman-equation.md` | Bellman equation, matrix-vector form, state/action values | Ch 2 |
| `03-bellman-optimality-equation.md` | BOE, optimal policy, contraction mapping | Ch 3 |
| `04-value-iteration-policy-iteration.md` | VI, PI, truncated PI, GPI pattern | Ch 4 |
| `05-monte-carlo-methods.md` | MC prediction, MC control, exploring starts, epsilon-greedy | Ch 5 |
| `06-stochastic-approximation.md` | Robbins-Monro, Dvoretzky, convergence conditions | Ch 6 |
| `07-temporal-difference-methods.md` | TD(0), Sarsa, Q-learning, n-step, convergence proofs | Ch 7 |
| `08-value-function-methods.md` | Linear FA, semi-gradient TD, DQN, deadly triad | Ch 8 |
| `09-policy-gradient-methods.md` | Policy gradient theorem, REINFORCE, baseline subtraction | Ch 9 |
| `10-actor-critic-methods.md` | QAC, A2C, off-policy AC, DPG, DDPG, TD3, SAC | Ch 10 |
| `11-appendix.md` | Mathematical prerequisites, norms, probability, linear algebra | Appendix |
| `12-cross-reference-index.md` | Algorithm comparison table, theorem index, symbol glossary | Ch 12 |

## Adding New Sources

To add a new source (book, paper, lecture notes):

1. Create a subdirectory under `knowledge-base/` named `<author>-<short-title>/`
2. Add markdown files following the numbered-prefix convention (`00-`, `01-`, ...)
3. Update this INDEX.md with the new source entry
4. Add the files to the `reference-files` list in `registry.json`

### Planned future sources
```
zhao-mathematical-foundations/    # Book 1 (current)
sutton-barto-2018/                # Book 2 (Sutton & Barto, 2nd ed.)
silver-lectures-2015/             # Lecture notes (David Silver UCL)
schulman-ppo-2017/                # Paper (Proximal Policy Optimization)
```
