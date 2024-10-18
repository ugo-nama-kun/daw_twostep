from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from two_step_env import DawTwoStepEnv, State, Action

env = DawTwoStepEnv()

N_EXPERIMENT = 10
N_TRIAL = 10_000

# model-free RL
# inverse temperature of decision
inverse_temp = 5.0
# discount rate
gamma = 1.0
# learning rate
lr = 0.5
# learning rate of models
lr_model = 0.5
# eligibility lambda
lambda_ = 1.0
# mb-mf weight (1.0 for full model-based)
w = 1.0

# stats
p_common_reward_list = np.zeros(N_EXPERIMENT)
p_rare_reward_list = np.zeros(N_EXPERIMENT)
p_common_unreward_list = np.zeros(N_EXPERIMENT)
p_rare_unreward_list = np.zeros(N_EXPERIMENT)

for n_experiment in range(N_EXPERIMENT):
    # models
    world_model = np.zeros((4, 2, 4))  # obs, action --> new_obs
    # state-action values
    Q_MF = np.zeros((4, 2))

    def get_action(obs: int):
        q_mf = Q_MF[obs]

        # model-based planning
        if obs != 0:
            q_mb = q_mf
        else:
            q_mb = np.zeros_like(q_mf)
            for a_ in Action:
                q_mb[a_.value] = 0
                for o_ in [State.s_A, State.s_B, State.s_C]:
                    q_mb[a_.value] += world_model[obs, a_.value, o_.value] * Q_MF[o_.value].max()

            # print(f"q_mb: {q_mb}")

        q_net = w * q_mb + (1 - w) * q_mf
        score = np.exp(inverse_temp * (q_net - q_net.max()))
        p_action = score / score.sum()

        if np.random.rand() < p_action[0]:
            action_ = 0  # Left
        else:
            action_ = 1  # Right

        return action_

    # Stay count
    common_reward = np.zeros(2)
    common_unreward = np.zeros(2)
    rare_reward = np.zeros(2)
    rare_unreward = np.zeros(2)

    # initialize latent state
    latent = np.random.randint(2)

    for trial in range(N_TRIAL):
        new_obs, info = env.reset()
        trajectory = []

        # Initial decision
        new_action = get_action(new_obs)

        trajectory.append((new_obs, new_action))

        # counting stay vs. non-stay
        if 1 < trial:
            # stay_counting
            if new_action == info["prev_action_at_state_A"]:
                # stay
                if info["prev_common_transition"]:
                    if info["prev_rewarding"]:
                        common_reward[0] += 1
                    else:
                        common_unreward[0] += 1
                else:
                    if info["prev_rewarding"]:
                        rare_reward[0] += 1
                    else:
                        rare_unreward[0] += 1
            else:
                # non-stay
                if info["prev_common_transition"]:
                    if info["prev_rewarding"]:
                        common_reward[1] += 1
                    else:
                        common_unreward[1] += 1
                else:
                    if info["prev_rewarding"]:
                        rare_reward[1] += 1
                    else:
                        rare_unreward[1] += 1

        done = False
        while not done:
            obs = deepcopy(new_obs)
            action = deepcopy(new_action)

            # transition
            new_obs, reward, done, truncated, info = env.step(action)

            # model learning
            # print(obs, action, new_obs)
            for o_ in [State.s_A, State.s_B, State.s_C]:
                x_ = float(new_obs == o_.value)
                world_model[obs, action, o_.value] = (1 - lr_model) * world_model[obs, action, o_.value] + lr_model * x_

            # SARSA(lambda)
            # next action
            new_action = get_action(new_obs)

            # for eligibility
            trajectory.append((new_obs, new_action))

            td_error = reward + gamma * (not done) * Q_MF[new_obs, new_action] - Q_MF[obs, action]

            eligibility = 1.0
            for obs_action in reversed(trajectory):
                Q_MF[obs_action[0], obs_action[1]] = Q_MF[obs_action[0], obs_action[1]] + lr * eligibility * td_error
                eligibility *= lambda_

            # if done:
            #     Q_MF[obs, action] = (1 - lr) * Q_MF[obs, action] + lr * reward

    p_common_reward_list[n_experiment] = common_reward[0] / common_reward.sum()
    p_rare_reward_list[n_experiment] = rare_reward[0] / rare_reward.sum()
    p_common_unreward_list[n_experiment] = common_unreward[0] / common_unreward.sum()
    p_rare_unreward_list[n_experiment] = rare_unreward[0] / rare_unreward.sum()

plt.figure()
plt.bar(
    ["common \n reward", "rare \n reward", "common \n unreward", "rare \n unreward"],
    [p_common_reward_list.mean(), p_rare_reward_list.mean(), p_common_unreward_list.mean(),
       p_rare_unreward_list.mean()],
    yerr=[p_common_reward_list.std(), p_rare_reward_list.std(), p_common_unreward_list.std(), p_rare_unreward_list.std()]
)
plt.ylim([0.5, 1])
plt.tight_layout()
plt.show()
