import gymnasium as gym
import mobile_env

print("[*] Start very large scenario...")

env = gym.make("mobile-verylarge-ma-v0", render_mode="rgb_array")

obs, info = env.reset()

print("[+] Number of InPs: ", env.NUM_InPs)
print("[+] Number of service providers: ", env.NUM_SPs)

print(
    f"\nVery large environment with:\
    \n* {env.NUM_USERS} users\
    \n* {env.NUM_STATIONS} base stations\
    \n* {env.NUM_EDGE_SERVERS} edge servers"
)

print("\n[*] InPs offered bundles:")
for inp in env.inps:
    print(f"     InP: {inp.inp_id}, number of edge servers: {inp.offer_bundle()}")

print("\n[*] Edge servers per InP:")
for inp in env.inps:
    print(f"     InP: {inp.inp_id}, number of edge servers: {len(inp.edge_servers)}")

print("\n[*] SPs' budgets and pricing:")
for sp in env.sps:
    print(
        f"     Initial budget: {sp.Budget} $, subscription fee: {sp.subscription_fee} $"
    )

print("\n[*] Users per SP:")
for sp in env.sps:
    print(f"     SP: {sp.sp_id}, number of users: {len(sp.users)}")

print("\n[*] Observation of SP 0:")
print(obs)
