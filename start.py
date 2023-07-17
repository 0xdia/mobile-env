import gymnasium as gym
import mobile_env


env = gym.make(
    "mobile-verylarge-ma-v0", render_mode="rgb_array", disable_env_checker=True
)


observations, info = env.reset()

# print(mobile_env.core.util.min_max_snr(env, observations))

done = False
iteration = 0
while not done:
    actions = {}
    for sp in env.sps:
        actions[sp.sp_id] = sp.action(observations[sp.sp_id])
    observations, rewards, terminated, truncated, info = env.step(
        actions
    )  # check step params

    iteration += 1
    done = terminated

env.close()
