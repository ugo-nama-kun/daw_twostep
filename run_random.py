

from two_step_env import DawTwoStepEnv

env = DawTwoStepEnv()

for trial in range(5):
    print(f"Trial #{trial}")
    obs, info = env.reset()
    print(f"initial obs: {obs}")
    print(f"initial info: {info}")
    print(f"rewarding: {info['prev_rewarding']}")
    print("")

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        print(f"obs: {obs}")
        print(f"reward: {reward}")
        print(f"done: {done}")
        # print(f"info: {info}")
        print("")

    print("---")

