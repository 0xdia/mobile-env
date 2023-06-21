import gymnasium as gym
import mobile_env
import time

env = gym.make("mobile-verylarge-ma-v0", render_mode="rgb_array", disable_env_checker=True)

start = time.time()

observations, info = env.reset()

# print(mobile_env.core.util.min_max_snr(env, observations))

done = False
while not done:
    actions = {}
    for sp in env.sps:
        actions[sp.sp_id] = sp.action(observations[sp.sp_id])
    observations2, rewards, terminated, truncated, info = env.step(
        actions
    )  # check step params
    done = terminated
    break

end = time.time()

# print("time = ", end - start)