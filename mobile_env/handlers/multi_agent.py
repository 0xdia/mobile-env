from typing import Dict

import gymnasium as gym
import numpy as np

from mobile_env.handlers.handler import Handler


class MComMAHandler(Handler):
    features = [
        "budget",
        "bundles",
        "tasks",
        "net-states",
    ]

    @classmethod
    def action_space(cls, env) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                ue.ue_id: gym.spaces.Discrete(env.NUM_STATIONS + 1)
                for ue in env.users.values()
            }
        )

    @classmethod
    def observation_space(cls, env) -> gym.spaces.Dict:
        bundle_space = gym.spaces.Dict(
            {
                "inp_id": gym.spaces.Discrete(env.NUM_InPs + 1),
                "bundle": gym.spaces.Dict(
                    {
                        "storage": gym.spaces.Discrete(100),
                        "vCPU": gym.spaces.Discrete(10),
                    }
                ),
            }
        )

        task_space = gym.spaces.Dict(
            {
                "ue_id": gym.spaces.Discrete(env.NUM_USERS + 1),
                "task": gym.spaces.Dict(
                    {
                        "compute": gym.spaces.Discrete(10),
                        "data": gym.spaces.Discrete(100),
                        "latency": gym.spaces.Discrete(100),
                    }
                ),
            }
        )

        net_states_space = gym.spaces.Dict(
            {
                "ue_id": gym.spaces.Discrete(100),
                "es_id": gym.spaces.Discrete(10),
                "snr": gym.spaces.Box(low=0.0, high=1e6, shape=(1,), dtype=float),
            }
        )

        space = gym.spaces.Dict(
            {
                "budget": gym.spaces.Box(low=0, high=1e5, shape=(1,), dtype=int),
                "bundles": gym.spaces.Tuple([bundle_space] * env.NUM_InPs),
                "tasks": gym.spaces.Tuple([task_space] * env.NUM_USERS),
                "net-states": gym.spaces.Tuple(
                    [net_states_space] * env.NUM_USERS * env.NUM_EDGE_SERVERS
                ),
            }
        )

        return space

    @classmethod
    def reward(cls, env):
        """UE's reward is their utility and the avg. utility of nearby BSs."""
        # compute average utility of UEs for each BS
        # set to lower bound if no UEs are connected
        bs_utilities = env.station_utilities()

        def ue_utility(ue):
            """Aggregates UE's own and nearby BSs' utility."""
            # ch eck what BS-UE connections are possible
            connectable = env.available_connections(ue)

            # utilities are broadcasted, i.e., aggregate utilities of BSs
            # that are in range of the UE
            ngbr_utility = sum(bs_utilities[bs] for bs in connectable)

            # calculate rewards as average weighted by
            # the number of each BSs' connections
            ngbr_counts = sum(len(env.connections[bs]) for bs in connectable)

            return (ngbr_utility + env.utilities[ue]) / (ngbr_counts + 1)

        rewards = {ue.ue_id: ue_utility(ue) for ue in env.active}
        return rewards

    @classmethod
    def observation(cls, env) -> Dict[int, np.ndarray]:
        """Select features for MA setting & flatten each UE's features."""

        # get features for currently active UEs
        """ active = set([ue.ue_id for ue in env.active if not env.done])
        features = env.features()
        features = {ue_id: obs for ue_id, obs in features.items() if ue_id in active} """

        # select observations for multi-agent setting from base feature set
        """ obs = {
            ue_id: [obs_dict[key] for key in cls.features]
            for ue_id, obs_dict in features.items()
        }
        """
        # flatten each UE's Dict observation to vector representation
        """ obs = {
            ue_id: np.concatenate([o for o in ue_obs]) for ue_id, ue_obs in obs.items()
        } """

        bundles_shape = (env.NUM_InPs, 3)
        tasks_shape = (env.NUM_USERS, 4)
        net_states_shape = (env.NUM_InPs * env.NUM_USERS, 3)

        for inp in env.inps:
            bundles = {
                "inp_id": inp.inp_id,
                "bundle": {
                    "storage": inp.bundle["storage"],
                    "vCPU": inp.bundle["vCPU"],
                },
            }

        observations = {
            sp.sp_id: {
                "budget": {sp.Budget},
                "bundles": bundles,
                "tasks": (),
                "net-states": (),
            }
            for sp in env.sps
        }

        print(observations)

        for inp in env.inps:
            bundles.append([inp.inp_id, inp.bundle["storage"], inp.bundle["vCPU"]])

        for sp in env.sps:
            for ue in sp.users:
                observations[sp.sp_id]["tasks"].append(
                    {
                        "ue_id": ue.ue_id,
                        "task": {
                            "compute": ue.task.compute_req,
                            "data": ue.task.data_req,
                            "latency": ue.task.latency_req,
                        },
                    }
                )

                for bs in env.stations:
                    observations[sp.sp_id]["net-states"].append(
                        {
                            "ue_id": ue.ue_id,
                            "task": {
                                "compute": ue.task.compute_req,
                                "data": ue.task.data_req,
                                "latency": ue.task.latency_req,
                            },
                        }
                    )

        for i in range(env.NUM_SPs):
            tasks = []
            for ue in env.users:
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
        for ue in env.users:
            if ue in env.sps[0].users:
                print("yes")
                for bs in env.stations:
                    net_states.append([ue.ue_id, bs.bs_id, env.channel.snr(bs, ue)])
        while len(net_states) < net_states_shape[0]:
            net_states.append([0.0, 0.0, 0.0])

        bundles = np.array(bundles, dtype=np.int32).reshape(bundles_shape)
        sp_0 = np.array(sp_observations[0], dtype=np.int32).reshape(tasks_shape)
        net_states = np.array(net_states, dtype=np.float64).reshape(net_states_shape)

        obs = {}
        return obs

    @classmethod
    def action(cls, env, action: Dict[int, int]):
        """Base environment by default expects action dictionary."""
        return action
