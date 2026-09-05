# Module 61: Reinforcement Learning Foundations

The mathematical core of learning from consequences: MDPs, Bellman
equations, Q-learning through PPO - with explicit connections to the
supervised ML you already know (and to RLHF for LLMs).

## What You Will Learn

- MDPs: states, actions, transitions, rewards, discounting
- Value functions V/Q and the Bellman equations (contraction view)
- TD learning, SARSA (on-policy), Q-learning (off-policy)
- DQN: experience replay + target networks (why each is needed)
- Policy gradients: the theorem, baselines, advantage
- Actor-critic and PPO (the practical standard; RLHF's engine)
- The ML connection table (DQN ~ regression, PG ~ weighted MLE)
- Debugging RL: reward hacking, deadly triad, sparse rewards

## Module Files

| File | Topic |
|------|-------|
| rl_foundations_complete.txt | Full foundations course with math |
| practice.txt | Exercises (12 items) |
| project.txt | Level 1-3 projects |
| think.txt | Hard reasoning incl. OPEN PROBLEM items |

## Prerequisites

- 06_ML_FUNDAMENTALS, 19_DEEP_LEARNING
- Probability + expectation comfort (05_MATHEMATICS/04)
- 28_LLM_FUNDAMENTALS helps for the RLHF application section

## Exit Criteria

- [ ] You can write the Bellman equations and explain the contraction
- [ ] You can explain SARSA vs Q-learning and why replay needs off-policy
- [ ] You can derive the policy gradient and explain the baseline
- [ ] You have implemented Q-learning or DQN on a small environment

## Interview Relevance

Exploration-exploitation, Bellman/DQN mechanics, policy gradient
intuition, PPO clip, RLHF - RL questions appear in senior ML and
alignment-focused roles.
