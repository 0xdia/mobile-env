import gymnasium as gym
import mobile_env


env = gym.make(
    "mobile-verylarge-ma-v0", render_mode="rgb_array", disable_env_checker=True
)


observation, info = env.reset()
done = False
while not done:
    action = env.sps[0].action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated

env.close()
