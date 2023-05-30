import random

import gymnasium as gym
import pandas

from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, EdgeServer, UserEquipment
from mobile_env.core.util import deep_dict_merge


class MComVeryLarge(MComCore):
    def __init__(self, config={}, render_mode=None):
        # set unspecified parameters to default configuration
        config = deep_dict_merge(self.default_config(), config)

        # the following code for station is temperory
        stations = [
            (50, 100),
            (60, 210),
            (90, 60),
            (120, 130),
            (130, 215),
            (140, 190),
            (160, 70),
            (200, 250),
            (210, 135),
            (230, 70),
            (250, 240),
            (255, 170),
            (265, 50),
        ]
        stations = [
            BaseStation(bs_id, pos, **config["bs"])
            for bs_id, pos in enumerate(stations)
        ]

        # @TODO: cluster edge server arounf base stations according to their locations
        df = pandas.read_csv(
            "~/repos/mobile-env/mobile_env/scenarios/very_large/site-optus-melbCBD.csv"
        ).iloc[1:, :3]
        edge_servers = [
            EdgeServer(
                df.iat[_, 0],
                0,
                random.randint(0, len(stations) - 1),
                df.iat[_, 1],
                df.iat[_, 2],
            )
            for _ in range(len(df))
        ]

        for es in edge_servers:
            stations[es.bs_id].add_edge_server(es.es_id)

        df = pandas.read_csv(
            "~/repos/mobile-env/mobile_env/scenarios/very_large/users-aus.csv"
        ).iloc[1:, 1:3]
        ues = [UserEquipment(ue_id, **config["ue"]) for ue_id in range(len(df))]

        super().__init__(stations, edge_servers, ues, config, render_mode)

        # @TODO: cleaning needed, verify this class and the mother class
        self.stations = stations
        self.edge_servers = edge_servers
        self.users = ues

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)
