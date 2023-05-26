from mobile_env.core.entities import BaseStation, ServiceProvider, EdgeInfrastructureProvider, EdgeServer, UserEquipment
from mobile_env.scenarios.verylarge import MComVeryLarge
import random


print("[*] Start very large scenario...")

scenario = MComVeryLarge()

TIME = 100000 # timestamps
NUM_SPs = 10 # service providers
NUM_InPs = 10 # edge infrastructure providers

sps = [ServiceProvider(_, random.randint(1000, 10000), 100, 50, random.randint(1, 5)) for _ in range(NUM_SPs)]
# attribute users to service providers
for user in scenario.users:
    sps[random.randint(0, NUM_SPs-1)].subscribe(user)

inps = [EdgeInfrastructureProvider(_) for _ in range(NUM_InPs)]
# attribute edge servers to inps
for es in scenario.edge_servers:
    random_inp = random.randint(0, NUM_InPs-1)
    es.inp = inps[random_inp]

# each edge server / inp offers a bundle
for es in scenario.edge_servers:
    es.offer_bundle()


print("[+] Number of InPs: ", NUM_InPs)
print("[+] Number of edge servers: ", scenario.NUM_EDGE_SERVERS)
print("[+] Number of base stations: ", scenario.NUM_STATIONS)
print("[+] Number of service providers: ", NUM_SPs)
print("[+] Number of users: ", scenario.NUM_USERS)

for t in range(TIME):
    # send tasks for corresponding service providers / network states
    # Sps bid for InPs
    pass