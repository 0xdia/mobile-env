import random
from collections import defaultdict
from typing import Dict

import gymnasium as gym
import numpy as np
import pandas
from numpy import unique
from sklearn.cluster import KMeans

from mobile_env.core.base import MComCoreMA
from mobile_env.core.entities import (
    BaseStation,
    EdgeInfrastructureProvider,
    EdgeServer,
    ServiceProvider,
    UserEquipment,
)
from mobile_env.core.util import deep_dict_merge

# np.set_printoptions(threshold=np.inf)


class MComVeryLarge(MComCoreMA):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)
        config["ue"]["velocity"] = 0
        num_of_bs = 13
        self.NUM_SPs = 10  # service providers
        self.NUM_InPs = 10  # edge infrastructure providers

        # @DONE: cluster edge server arounf base stations according to their locations
        df = pandas.read_csv(
            "~/repos/mobile-env/mobile_env/scenarios/very_large/site-optus-melbCBD.csv"
        ).iloc[1:, 1:3]
        edge_servers = [
            EdgeServer(
                _,
                0,
                df.iat[_, 0],
                df.iat[_, 1],
            )
            for _ in range(len(df))
        ]

        self.inps = [EdgeInfrastructureProvider(_) for _ in range(self.NUM_InPs)]

        # determining base station locations
        df = pandas.DataFrame(df).to_numpy()
        model = KMeans(n_clusters=num_of_bs)
        model.fit(df)
        clustring = model.predict(df)

        stations = [
            BaseStation(
                _,
                (model.cluster_centers_[_, 0], model.cluster_centers_[_, 1]),
                **config["bs"],
            )
            for _ in range(num_of_bs)
        ]

        # attribute edge servers to base stations
        for i in range(len(edge_servers)):
            edge_servers[i].bs_id = clustring[i]
            stations[clustring[i]].add_edge_server(edge_servers[i])

        df = pandas.read_csv(
            "~/repos/mobile-env/mobile_env/scenarios/very_large/users-melbcbd-generated.csv"
        ).iloc[1:, 0:3]
        self.ues = []
        for _ in range(len(df)):
            ue = UserEquipment(_, **config["ue"])
            ue.x, ue.y = df.iat[_, 0], df.iat[_, 1]
            self.ues.append(ue)

        self.sps = [
            ServiceProvider(
                _, random.randint(1000, 10000), 100, 50, random.randint(1, 5)
            )
            for _ in range(self.NUM_SPs)
        ]

        super().__init__(stations, edge_servers, self.ues, config, render_mode)

        # @TODO: cleaning needed, verify this class and the mother class
        self.stations = stations
        self.edge_servers = edge_servers
        self.users = self.ues

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        # reset time
        self.time = 0.0
        # set seed
        if seed is not None:
            self.seeding({"seed": seed})

        # initialize RNG or reset (if necessary on episode end)
        if self.reset_rng_episode or self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        # reset state kept by arrival pattern, channel, scheduler, etc.
        self.arrival.reset()
        self.channel.reset()
        self.scheduler.reset()
        self.movement.reset()
        self.utility.reset()

        # generate new arrival and exit times for UEs
        for ue in self.users:
            ue.stime = self.arrival.arrival(ue)
            ue.extime = self.arrival.departure(ue)

        # initially not all UEs request downlink connections (service)
        self.active = [ue for ue in self.users if ue.stime <= 0]
        self.active = sorted(self.active, key=lambda ue: ue.ue_id)

        # reset established downlink connections (default empty set)
        self.connections = defaultdict(set)
        # reset connections' data rates (defaults set to 0.0)
        self.datarates = defaultdict(float)
        # reset UEs' utilities
        self.utilities = {}

        # set time of last UE's departure
        self.max_departure = max(ue.extime for ue in self.users)

        # reset episode's results of metrics tracked by the monitor
        self.monitor.reset()

        # check if handler is applicable to mobile scenario
        # NOTE: e.g. fails if the central handler is used,
        # although the number of UEs changes
        self.handler.check(self)

        # info
        #info = self.handler.info(self)
        # store latest monitored results in `info` dictionary
        #info = {**info, **self.monitor.info()}

        # attribute edge servers to inps
        for es in self.edge_servers:
            random_inp = random.randint(0, self.NUM_InPs - 1)
            es.inp = self.inps[random_inp]
            self.inps[random_inp].edge_servers.append(es)

        # each edge server / inp offers a bundle
        for es in self.edge_servers:
            es.offer_bundle()

        # attribute users to service providers
        for user in self.users:
            random_sp = random.randint(0, self.NUM_SPs - 1)
            self.sps[random_sp].subscribe(user)
            user.current_sp = random_sp

        # generate tasks on UEs
        for ue in self.users:
            ue.generate_task()

        return self.handler.observation(self), {}

    def step(self, actions: Dict[int, int]):
        assert not self.time_is_up, "step() called on terminated episode"

        # release established connections that moved e.g. out-of-range
        self.update_connections()

        # TODO: add penalties for changing connections?
        for sp_id, action in actions.items():
            self.apply_action(action, sp_id)

        # InPs decides bidding war winners
        for inp in self.inps:
            winner = inp.decide_bidding_winner()
            self.sps[winner[0]].pay(inp.inp_id, winner[1])

        # update connections' data rates after re-scheduling
        self.datarates = {}
        for bs in self.stations:
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update macro (aggregated) data rates for each UE
        self.macro = self.macro_datarates(self.datarates)

        # compute utilities from UEs' data rates & log its mean value
        self.utilities = {
            ue: self.utility.utility(self.macro[ue]) for ue in self.active
        }

        # scale utilities to range [-1, 1] before computing rewards
        self.utilities = {
            ue: self.utility.scale(util) for ue, util in self.utilities.items()
        }

        rewards = self.handler.reward(self)

        # evaluate metrics and update tracked metrics given the core simulation
        self.monitor.update(self)

        # move user equipments around; update positions of UEs
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # terminate existing connections for exiting UEs
        leaving = set([ue for ue in self.active if ue.extime <= self.time])
        for bs, ues in self.connections.items():
            self.connections[bs] = ues - leaving

        # update list of active UEs & add those that begin to request service
        self.active = sorted(
            [
                ue
                for ue in self.users
                if ue.extime > self.time and ue.stime <= self.time
            ],
            key=lambda ue: ue.ue_id,
        )

        # update the data rate of each (BS, UE) connection after movement
        for bs in self.stations:
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update internal time of environment
        self.time += 1

        # check whether episode is done & close the environment
        if self.time_is_up and self.window:
            self.close()

        # do not invoke next step on policies before at least one UE is active
        if not self.active and not self.time_is_up:
            return self.step({})

        # compute observations for next step and information
        # methods are defined by handler according to strategy pattern
        # NOTE: compute observations after proceeding in time (may skip ahead)
        observation = self.handler.observation(self)
        
        # info = self.handler.info(self)
        # store latest monitored results in `info` dictionary
        # info = {**info, **self.monitor.info()}

        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        terminated = False
        truncated = self.time_is_up

        return observation, rewards, terminated, truncated, {}

    def apply_action(self, action: Dict[int, int], sp_id: int) -> None:
        for inp, bid in action.items():
            self.inps[inp].receive_bid(sp_id, bid)