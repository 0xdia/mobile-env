import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pandas
from numpy import unique
from sklearn.cluster import KMeans

from mobile_env.core.base import MComCore
from mobile_env.core.entities import (
    BaseStation,
    EdgeServer,
    ServiceProvider,
    UserEquipment,
    EdgeInfrastructureProvider,
)
from mobile_env.core.util import deep_dict_merge


class MComVeryLarge(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)
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

        inps = [EdgeInfrastructureProvider(_) for _ in range(self.NUM_InPs)]
        # attribute edge servers to inps
        for es in edge_servers:
            random_inp = random.randint(0, self.NUM_InPs - 1)
            es.inp = inps[random_inp]

        # each edge server / inp offers a bundle
        for es in edge_servers:
            es.offer_bundle()

        # determining base station locations
        df = pandas.DataFrame(df).to_numpy()
        model = KMeans(n_clusters=num_of_bs)
        model.fit(df)
        clustring = model.predict(df)

        stations = [
            BaseStation(
                _,
                (model.cluster_centers_[_, 0], model.cluster_centers_[_, 1]),
                **config["bs"]
            )
            for _ in range(num_of_bs)
        ]

        # attribute edge servers to base stations
        for i in range(len(edge_servers)):
            edge_servers[i].bs_id = clustring[i]
            stations[clustring[i]].add_edge_server(edge_servers[i].es_id)

        df = pandas.read_csv(
            "~/repos/mobile-env/mobile_env/scenarios/very_large/users-aus.csv"
        ).iloc[1:, 1:3]
        ues = []
        for _ in range(len(df)):
            ue = UserEquipment(_, **config["ue"])
            ue.x, ue.y = df.iat[_, 0], df.iat[_, 1]
            ues.append(ue)

        sps = [
            ServiceProvider(
                _, random.randint(1000, 10000), 100, 50, random.randint(1, 5)
            )
            for _ in range(self.NUM_SPs)
        ]
        # attribute users to service providers
        for user in ues:
            sps[random.randint(0, self.NUM_SPs - 1)].subscribe(user)

        super().__init__(stations, edge_servers, ues, config, render_mode)

        # @TODO: cleaning needed, verify this class and the mother class
        self.stations = stations
        self.edge_servers = edge_servers
        self.users = ues

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

        # extra options currently not supported
        if options is not None:
            raise NotImplementedError(
                "Passing extra options on env.reset() is not supported."
            )

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

        return self.handler.observation(self), info
