from mobile_env.core.entities import BaseStation, EdgeInfrastructureProvider, EdgeServer, UserEquipment
from mobile_env.scenarios.verylarge import MComVeryLarge
import random

scenario = MComVeryLarge()

TIME = 100000 # timestamps
NUM_SPs = 100 # service providers
NUM_InPs = 130 # edge infrastructure providers

sps = [[] for _ in range(NUM_SPs)]
# attribute users to service providers
for user in scenario.users:
    sps[random.randint(0, NUM_SPs-1)].append(user.ue_id)

inps = [EdgeInfrastructureProvider(_) for _ in range(NUM_InPs)]
# attribute edge servers to inps
for es in scenario.edge_servers:
    random_inp = random.randint(0, NUM_InPs-1)
    es.inp = inps[random_inp]

# each edge server / inp offers a bundle
for es in scenario.edge_servers:
    es.offer_bundle()


for t in range(TIME):
    
    # send tasks for corresponding service providers / network states
    # sps bid for InPs
    pass