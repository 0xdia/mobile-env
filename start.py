import gymnasium as gym
import mobile_env
import time
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

env = gym.make(
    "mobile-verylarge-ma-v0", render_mode="rgb_array", disable_env_checker=True
)

start = time.time()

observations, info = env.reset()

# print(mobile_env.core.util.min_max_snr(env, observations))

done = False
iter = 0
while not done:
    actions = {}
    for sp in env.sps:
        actions[sp.sp_id] = sp.action(observations[sp.sp_id])
    observations, rewards, terminated, truncated, info = env.step(
        actions
    )  # check step params

    writer.add_scalars(
        "randomness",
        {"sp_" + str(sp.sp_id): sp.Budget for sp in env.sps},
        iter,
    )
    iter += 1
    done = terminated

end = time.time()

writer.close()
# print("time = ", end - start)
