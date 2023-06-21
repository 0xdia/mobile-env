import gymnasium as gym
import mobile_env

env = gym.make("mobile-verylarge-ma-v0", render_mode="rgb_array")

observations, info = env.reset()

done = False
while not done:
    actions = {}
    for sp in env.sps:
        actions[sp.sp_id] = sp.action(observations[sp.sp_id])
    observation, rewards, terminated, truncated, info = env.step(
        actions
    )  # check step params
    print(rewards)
    done = terminated