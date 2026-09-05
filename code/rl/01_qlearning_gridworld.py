"""
Q-learning on a gridworld with a stochastic "slip" - runnable RL example.
Implements the tabular algorithms of 61_REINFORCEMENT_LEARNING and
verifies the learned policy against exact value iteration.

Run:  python code/rl/01_qlearning_gridworld.py
Expected: Q-learning reaches the goal; its greedy policy is near-optimal
despite the 15% slip noise (the lesson's exercise #11 in code).
"""
import numpy as np


class GridWorld:
    """4x5 grid: start (0,0), goal (4,2) reward +10, pit (4,4) reward -10.
    Actions N/S/E/W; the chosen move executes with 85% probability and
    each perpendicular move with 5% (stochastic transitions)."""

    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # E, W, S, N
    SIZE = (5, 5)   # rows 0..4, cols 0..4 - goal/pit must be ON the grid
    START, GOAL, PIT = (0, 0), (4, 2), (4, 4)
    SLIP = 0.15

    def reset(self):
        return self.START

    def transitions(self, s, a):
        """DETERMINISTIC list of (probability, next_state, reward, done)
        for taking action a in state s - the environment's expectation,
        used by value iteration (never sample randomness inside a DP)."""
        if s == self.GOAL or s == self.PIT:
            return [(1.0, s, 0.0, True)]
        out = []
        for m in range(4):
            p = (1.0 - self.SLIP) if m == a else (self.SLIP / 3.0)
            dr, dc = self.ACTIONS[m]
            r, c = s[0] + dr, s[1] + dc
            r = max(0, min(self.SIZE[0] - 1, r))
            c = max(0, min(self.SIZE[1] - 1, c))
            ns = (r, c)
            rew = 10.0 if ns == self.GOAL else -10.0 if ns == self.PIT else -0.1
            out.append((p, ns, rew, ns in (self.GOAL, self.PIT)))
        return out

    def step(self, s, a):
        """Stochastic sampler for LEARNING (Q-learning)."""
        if s == self.GOAL or s == self.PIT:
            return s, 0.0, True
        a_actual = a if np.random.random() > self.SLIP \
            else np.random.choice([m for m in range(4) if m != a])
        out = [t for t in self.transitions(s, a_actual) if t[1] != s]
        # pick the executed branch (transitions already returned one per m)
        dr, dc = self.ACTIONS[a_actual]
        r, c = max(0, min(self.SIZE[0] - 1, s[0] + dr)), \
               max(0, min(self.SIZE[1] - 1, s[1] + dc))
        ns = (r, c)
        rew = 10.0 if ns == self.GOAL else -10.0 if ns == self.PIT else -0.1
        return ns, rew, ns in (self.GOAL, self.PIT)


def value_iteration(env, gamma=0.95, eps=1e-9):
    """Exact optimal value function (Bellman optimality iteration)."""
    V = np.zeros(env.SIZE)
    while True:
        delta = 0.0
        for r in range(env.SIZE[0]):
            for c in range(env.SIZE[1]):
                s = (r, c)
                if s in (env.GOAL, env.PIT):
                    continue
                best = max(
                    sum(p * (rew + gamma * V[ns])
                        for p, ns, rew, _ in env.transitions(s, a))
                    for a in range(4))
                delta = max(delta, abs(V[r, c] - best))
                V[r, c] = best
        if delta < eps:
            break
    return V


def q_learning(env, gamma=0.95, lr=0.1, episodes=6000, eps0=0.9,
               eps_min=0.05, decay=0.998, max_steps=200):
    Q = np.zeros((*env.SIZE, 4))
    eps = eps0
    for _ in range(episodes):
        s = env.reset()
        done = False
        for _ in range(max_steps):
            a = np.argmax(Q[s]) if np.random.random() > eps \
                else np.random.randint(4)
            ns, r, done = env.step(s, a)
            Q[s + (a,)] += lr * (r + gamma * Q[ns].max() - Q[s + (a,)])
            s = ns
            if done:
                break
        eps = max(eps_min, eps * decay)
    return Q


def greedy_policy(Q):
    return np.array([[int(np.argmax(Q[r, c])) for c in range(Q.shape[1])]
                     for r in range(Q.shape[0])])


def optimal_policy(env, V, gamma=0.95):
    pi = np.zeros(env.SIZE, dtype=int)
    for r in range(env.SIZE[0]):
        for c in range(env.SIZE[1]):
            s = (r, c)
            if s in (env.GOAL, env.PIT):
                continue
            pi[s] = int(max(
                range(4),
                key=lambda a: sum(p * (rew + gamma * V[ns])
                                  for p, ns, rew, _ in env.transitions(s, a))))
    return pi


def policy_return(env, policy, trials=3000, max_steps=100):
    """Mean discounted return of a deterministic policy (Monte Carlo)."""
    gammas = 0.95 ** np.arange(max_steps)
    returns = []
    for _ in range(trials):
        s, done, rew = env.reset(), False, []
        for _ in range(max_steps):
            s, r, done = env.step(s, int(policy[s]))
            rew.append(r)
            if done:
                break
        returns.append(float(np.dot(gammas[:len(rew)], rew)))
    return float(np.mean(returns))


def main():
    np.random.seed(1)
    env = GridWorld()
    V_star = value_iteration(env)
    Q = q_learning(env)
    pi_ql = greedy_policy(Q)
    pi_star = optimal_policy(env, V_star)
    print("learned greedy policy (row 0 = top; N/S/E/W encoded 0-3):")
    print(pi_ql)
    g_star = policy_return(env, pi_star)
    g_ql = policy_return(env, pi_ql)
    print(f"\noptimal policy  mean discounted return: {g_star:.3f}")
    print(f"Q-learning      mean discounted return: {g_ql:.3f}")
    print("=> PASS: Q-learning reached near-optimal return under slip noise"
          if g_ql > g_star * 0.9 else "=> tune episodes/lr/eps decay")


if __name__ == "__main__":
    main()
