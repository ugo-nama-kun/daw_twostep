from copy import deepcopy

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from enum import Enum


class State(Enum):
    s_A = 0
    s_B = 1
    s_C = 2
    Terminal = 3


class Action(Enum):
    a_A = 0
    a_B = 1


class DawTwoStepEnv(gym.Env):

    def __init__(self):
        super(DawTwoStepEnv, self).__init__()
        # left or right
        self.action_space = spaces.Discrete(2)
        # s_A, s_B, s_C
        self.observation_space = spaces.Discrete(3)

        self.state = None
        self.t = None

        self.prob_transition_common = 0.7

        self.prob_low = 0.25
        self.prob_high = 0.75
        self.prob_drift_scale = 0.025

        # Initial reward probabilities
        self.prob_reward_B = np.random.uniform(low=0.25, high=0.75, size=2)  # rewarding prob for left, right actions
        self.prob_reward_C = np.random.uniform(low=0.25, high=0.75, size=2)

        self.prev_common_transition = False
        self.prev_action_at_state_A = None
        self.prev_rewarding = False

    def step(self, action: int):
        assert action in [Action.a_A.value, Action.a_B.value]

        reward = 0.0
        self.t += 1
        info = {
            "prob_reward_B": np.array(self.prob_reward_B, dtype=np.float32),
            "prob_reward_C": np.array(self.prob_reward_C, dtype=np.float32),
        }

        # state transition
        if self.state is State.s_A:  # Choice State
            self.prev_action_at_state_A = deepcopy(action)
            reward = 0.0
            if action == Action.a_A.value:
                if np.random.rand() < self.prob_transition_common:
                    self.state = State.s_B
                    self.prev_common_transition = True
                else:
                    self.state = State.s_C
            elif action == Action.a_B.value:
                if np.random.rand() < self.prob_transition_common:
                    self.state = State.s_C
                    self.prev_common_transition = True
                else:
                    self.state = State.s_B

        elif self.state is State.s_B:  # State B
            self.state = State.Terminal
            if action == Action.a_A.value:
                if np.random.rand() < self.prob_reward_B[0]:
                    reward = 1.0
                    self.prev_rewarding = True
            elif action == Action.a_B.value:
                if np.random.rand() < self.prob_reward_B[1]:
                    reward = 1.0
                    self.prev_rewarding = True

        elif self.state is State.s_C:  # State C
            self.state = State.Terminal
            if action == Action.a_A.value:
                if np.random.rand() < self.prob_reward_C[0]:
                    reward = 1.0
                    self.prev_rewarding = True
            elif action == Action.a_B.value:
                if np.random.rand() < self.prob_reward_C[1]:
                    reward = 1.0
                    self.prev_rewarding = True

        observation = self.state.value
        terminated = self.t == 2
        truncated = False

        # updating reward probs at the end of each trial
        if terminated:
            for i in range(len(Action)):
                self.prob_reward_B[i] = np.clip(
                    self.prob_reward_B[i] + self.prob_drift_scale * np.random.randn(),
                    a_min=self.prob_low,
                    a_max=self.prob_high
                )
                self.prob_reward_C[i] = np.clip(
                    self.prob_reward_C[i] + self.prob_drift_scale * np.random.randn(),
                    a_min=self.prob_low,
                    a_max=self.prob_high
                )

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.t = 0
        self.state = State.s_A
        observation = self.state.value
        info = {
            "prob_reward_B": np.array(self.prob_reward_B, dtype=np.float32),
            "prob_reward_C": np.array(self.prob_reward_C, dtype=np.float32),
            "prev_common_transition": self.prev_common_transition,
            "prev_rewarding": self.prev_rewarding,
            "prev_action_at_state_A": self.prev_action_at_state_A,
        }

        self.prev_common_transition = False
        self.prev_rewarding = False

        return observation, info
