from mobile_env.core.entities import BaseStation, EdgeServer, UserEquipment
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

inps = [0 for _ in range(NUM_InPs)]
# attribute edge servers to inps
for es in scenario.edge_servers:
    random_inp = random.randint(0, NUM_InPs-1)
    es.inp_id = random_inp
    inps[random_inp] += 1

for t in range(TIME):
    # generate tasks from users
    # send tasks for corresponding service providers / network states
    # send bundle offers from inps for sps
    # sps bid for InPs
    pass