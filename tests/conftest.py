"""Shared fixtures for rl-methodology validation tests."""
import numpy as np
import pytest


class GridWorld:
    """3x3 grid world environment from rl-methodology SKILL.md."""

    def __init__(self, size=3, gamma=0.9):
        self.size = size
        self.n_states = size * size
        self.n_actions = 5  # up, down, left, right, stay
        self.gamma = gamma
        self.action_effects = {
            0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)
        }
        self.forbidden_states = set()
        self.target_state = None
        self.rewards = {}

    def state_to_rc(self, s):
        return s // self.size, s % self.size

    def rc_to_state(self, r, c):
        return r * self.size + c

    def step(self, state, action):
        r, c = self.state_to_rc(state)
        dr, dc = self.action_effects[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            next_state = self.rc_to_state(nr, nc)
            if next_state not in self.forbidden_states:
                reward = self.rewards.get((state, action, next_state), -1)
                return next_state, reward
        reward = self.rewards.get((state, action, state), -1)
        return state, reward

    def get_transition_model(self):
        """Returns p(s'|s,a) as array of shape (n_states, n_actions, n_states)."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                s_next, _ = self.step(s, a)
                P[s, a, s_next] = 1.0
        return P

    def get_reward_vector(self):
        """Returns r(s,a) as array of shape (n_states, n_actions)."""
        R = np.full((self.n_states, self.n_actions), -1.0)
        for (s, a, s_next), r in self.rewards.items():
            R[s, a] = r
        return R


@pytest.fixture
def env():
    """Standard 3x3 GridWorld with target at state 8 (bottom-right)."""
    gw = GridWorld(size=3, gamma=0.9)
    gw.target_state = 8
    gw.rewards[(7, 3, 8)] = 0  # reaching goal from state 7, action right
    gw.rewards[(5, 1, 8)] = 0  # reaching goal from state 5, action down
    return gw


@pytest.fixture
def env_plain():
    """Plain 3x3 GridWorld, reward -1 everywhere, no target."""
    return GridWorld(size=3, gamma=0.9)
