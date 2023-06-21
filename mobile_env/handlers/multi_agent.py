from typing import Dict

import gymnasium as gym
import numpy as np

from mobile_env.handlers.handler import Handler

from collections import OrderedDict

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
                "budget": gym.spaces.Discrete(100001),
                "bundles": bundles_space,
                "tasks": tasks_space,
                "net-states": net_states_space,
            }
        )

        space = gym.spaces.Dict({sp.sp_id: sp_space for sp in env.sps})
        cls.space = space
        return space

    @classmethod
    def reward(cls, env):
        rewards = {sp.sp_id: 1 for sp in env.sps}
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

        observations = OrderedDict({
            sp.sp_id: {
                "budget": sp.Budget,
                "bundles": np.array(bundles),
                "tasks": [[0 for _ in range(4)] for _ in range(env.NUM_USERS)],
                "net-states": [],
            }
            for sp in env.sps
        })

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

        for sp in observations:
            observations[sp]["tasks"] = np.array(observations[sp]["tasks"], dtype=np.int32)
            observations[sp]["net-states"] = np.array(observations[sp]["net-states"])

        return observations

    @classmethod
    def action(cls, env, action: Dict[int, int]):
        """Base environment by default expects action dictionary."""
        return action
