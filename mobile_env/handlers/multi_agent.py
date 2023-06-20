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
        sp_action_space = gym.spaces.Dict(
            {inp.inp_id: gym.spaces.Discrete(10001) for inp in env.inps}
        )
        return gym.spaces.Dict({sp.sp_id: sp_action_space for sp in env.sps})

    @classmethod
    def observation_space(cls, env) -> gym.spaces.Dict:
        bundles_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(env.NUM_InPs, 3), dtype=np.int32
        )
        tasks_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(env.NUM_USERS, 4), dtype=np.int32
        )
        net_states_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=(env.NUM_USERS * env.NUM_EDGE_SERVERS, 3),
            dtype=np.float64,
        )

        sp_space = gym.spaces.Dict(
            {
                "budget": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int),
                "bundles": bundles_space,
                "tasks": tasks_space,
                "net-states": net_states_space,
            }
        )

        space = {sp.sp_id: sp_space for sp in env.sps}

        space = gym.spaces.Dict(space)
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

        bundles = []
        for inp in env.inps:
            bundles.append([inp.inp_id, inp.bundle["storage"], inp.bundle["vCPU"]])

        observations = {
            sp.sp_id: {
                "budget": {sp.Budget},
                "bundles": bundles,
                "tasks": [[0 for _ in range(4)] for _ in range(env.NUM_USERS)],
                "net-states": [],
            }
            for sp in env.sps
        }

        for ue in env.users:
            observations[ue.current_sp]["tasks"][ue.ue_id] = [
                ue.ue_id,
                ue.task.computing_req,
                ue.task.data_req,
                ue.task.latency_req,
            ]

        for ue in env.users:
            for es in env.edge_servers:
                observations[ue.current_sp]["net-states"].append(
                    [ue.ue_id, es.es_id, env.channel.snr(env.stations[es.bs_id], ue)]
                )

        print(observations[1]["budget"])
        return observations

    @classmethod
    def action(cls, env, action: Dict[int, int]):
        """Base environment by default expects action dictionary."""
        return action
