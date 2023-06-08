import gymnasium as gym
import mobile_env

print("[*] Start very large scenario...")

env = gym.make("mobile-verylarge-central-v0", render_mode="rgb_array")


print("[+] Number of InPs: ", env.NUM_InPs)
print("[+] Number of service providers: ", env.NUM_SPs)

print(
    f"Very large environment with:\
    \n* {env.NUM_USERS} users\
    \n* {env.NUM_STATIONS} base stations\
    \n* {env.NUM_EDGE_SERVERS} edge servers"
)

for bs in env.stations:
    for es in env.edge_servers:
        continue

obs, info = env.reset()

TIME = 100000  # timestamps
for t in range(TIME):
    # send tasks for corresponding service providers / network states
    # Sps bid for InPs ==> action space
    dummy_action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(dummy_action)
