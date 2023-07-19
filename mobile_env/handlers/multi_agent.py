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
        "net-states",  # https://support.zyxel.eu/hc/en-us/articles/4406391493778-5G-signal-quality-parameters
    ]

    @classmethod
    def action_space(cls, env):
        sp_action_space = gym.spaces.Box(
            low=0, high=10001, shape=(env.NUM_InPs,), dtype=np.int32
        )
        return sp_action_space

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

        # space = gym.spaces.Dict({sp.sp_id: sp_space for sp in env.sps})
        cls.space = sp_space
        return sp_space

    @classmethod
    def reward(cls, env):
        cls.rewards = {}
        for sp in env.sps:
            cls.rewards[sp.sp_id] = 0.1 * sp.bids_won / env.NUM_SPs - 0.9 * sp.last_spending / 400
            sp.bids_won = sp.last_spending = 0
        external_agent_reward = cls.rewards[-1]
        return external_agent_reward

    @classmethod
    def observation(cls, env):
        bundles = []
        for inp in env.inps:
            bundles.append([inp.inp_id, inp.bundle["storage"], inp.bundle["vCPU"]])

        observations = OrderedDict(
            {
                sp.sp_id: {
                    "budget": sp.Budget,
                    "bundles": np.array(bundles),
                    "tasks": [[0 for _ in range(4)] for _ in range(env.NUM_USERS)],
                    "net-states": [
                        [
                            ue.ue_id,
                            es.es_id,
                            env.channel.snr(env.stations[es.bs_id], ue),
                        ]
                        for ue in env.users
                        for es in env.edge_servers
                    ],
                }
                for sp in env.sps
            }
        )

        for ue in env.users:
            observations[ue.current_sp]["tasks"][ue.ue_id] = [
                ue.ue_id,
                ue.task.computing_req,
                ue.task.data_req,
                ue.task.latency_req,
            ]

        """ for ue in env.users:
            for es in env.edge_servers:
                observations[ue.current_sp]["net-states"].append(
                    [ue.ue_id, es.es_id, env.channel.snr(env.stations[es.bs_id], ue)]
                ) """

        for sp in observations:
            observations[sp]["tasks"] = np.array(
                observations[sp]["tasks"], dtype=np.int32
            )
            observations[sp]["net-states"] = np.array(observations[sp]["net-states"])

        external_agent_observation = observations[-1].copy()
        observations.pop(-1)
        env.internal_agents_observations = observations.copy()
        return external_agent_observation

    @classmethod
    def action(cls, env, action: Dict[int, int]):
        """Base environment by default expects action dictionary."""
        return action
