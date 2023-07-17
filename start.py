import gymnasium as gym
import mobile_env


env = gym.make(
    "mobile-verylarge-ma-v0", render_mode="rgb_array", disable_env_checker=True
)


observations, info = env.reset()

# print(mobile_env.core.util.min_max_snr(env, observations))

done = False
while not done:
    actions = {}
    for sp in env.sps:
        actions[sp.sp_id] = sp.action(observations[sp.sp_id])
    observations, rewards, terminated, truncated, info = env.step(
        actions
    ) 
    done = terminated

env.close()
