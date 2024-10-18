import numpy as np
import matplotlib.pyplot as plt

from two_step_env import DawTwoStepEnv

env = DawTwoStepEnv()

N_TRIAL = 10_000

# Stay count
common_reward = np.zeros(2)
common_unreward = np.zeros(2)
rare_reward = np.zeros(2)
rare_unreward = np.zeros(2)

for trial in range(N_TRIAL):
    obs, info = env.reset()

    # Initial decision
    action = env.action_space.sample()

    # counting stay vs. non-stay
    if 1 < trial:
        # stay_counting
        if action == info["prev_action_at_state_A"]:
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
        obs, reward, done, truncated, info = env.step(action)

p_common_reward = common_reward[0] / common_reward.sum()
p_rare_reward = rare_reward[0] / rare_reward.sum()
p_common_unreward = common_unreward[0] / common_unreward.sum()
p_rare_unreward = rare_unreward[0] / rare_unreward.sum()
p = np.array([p_common_reward, p_rare_reward, p_common_unreward, p_rare_unreward])
print(p)

plt.figure()
plt.bar(
    ["common \n reward", "rare \n reward", "common \n unreward", "rare \n unreward"],
    p
)
plt.ylim([0, 1])
plt.tight_layout()
plt.show()
