import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pandas
from numpy import unique
from sklearn.cluster import KMeans

from mobile_env.core.base import MComCoreMA
from mobile_env.core.entities import (
    BaseStation,
    EdgeServer,
    ServiceProvider,
    UserEquipment,
    EdgeInfrastructureProvider,
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
        info = self.handler.info(self)
        # store latest monitored results in `info` dictionary
        info = {**info, **self.monitor.info()}

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

        return self.handler.observation(self), info

        bundles_shape = (10, 3)
        tasks_shape = (815, 4)
        net_states_shape = (815 * 124, 3)

        bundles = []
        for inp in self.inps:
            bundles.append([inp.inp_id, inp.bundle["storage"], inp.bundle["vCPU"]])

        sp_observations = {}
        for i in range(self.NUM_SPs):
            tasks = []
            for ue in self.users:
                if ue.current_sp == i:
                    tasks.append(
                        [
                            ue.ue_id,
                            ue.task.computing_req,
                            ue.task.data_req,
                            ue.task.latency_req,
                        ]
                    )

            while len(tasks) < tasks_shape[0]:
                tasks.append([0, 0, 0, 0])
            sp_observations[i] = tasks

        net_states = []
        for ue in self.users:
            if ue in self.sps[0].users:
                print("yes")
                for bs in self.stations:
                    net_states.append([ue.ue_id, bs.bs_id, self.channel.snr(bs, ue)])
        while len(net_states) < net_states_shape[0]:
            net_states.append([0.0, 0.0, 0.0])

        bundles = np.array(bundles, dtype=np.int32).reshape(bundles_shape)
        sp_0 = np.array(sp_observations[0], dtype=np.int32).reshape(tasks_shape)
        net_states = np.array(net_states, dtype=np.float64).reshape(net_states_shape)

        observation = {"bundles": bundles, "tasks": sp_0, "net-states": net_states}

        return observation, info
