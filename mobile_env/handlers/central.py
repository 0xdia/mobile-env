from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

from mobile_env.handlers.handler import Handler


class MComCentralHandler(Handler):
    features = ["connections", "snrs", "utility"]

    @classmethod
    def ue_obs_size(cls, env) -> int:
        return sum(env.feature_sizes[ftr] for ftr in cls.features)

    @classmethod
    def action_space(cls, env) -> spaces.MultiDiscrete:
        # define multi-discrete action space for central setting
        # each element of a multi-discrete action denotes one UE's decision
        # TODO: 1. pay attention to the four loop
        #       2. change the action space
        return spaces.MultiDiscrete([env.NUM_STATIONS + 1 for _ in env.users])

    @classmethod
    def observation_space(cls, env) -> spaces.Box:
        # look at the overleaf presentation for details about the state space.
        # size = cls.ue_obs_size(env)
        # return spaces.Box(low=-1.0, high=1.0, shape=(env.NUM_USERS * size,))
        return spaces.Dict(
            {
                "bundles": spaces.Box(low=0, high=99999, shape=(10, 3), dtype=np.int32),
                "tasks": spaces.Box(low=0, high=850, shape=(815, 4), dtype=np.int32),
                "net-states": spaces.Box(
                    low=0.0, high=500000.0, shape=(815 * 124, 3), dtype=np.float32
                ),
            }
        )

    @classmethod
    def action(cls, env, actions: Tuple[int]) -> Dict[int, int]:
        """Transform flattend actions to expected shape of core environment."""
        assert len(actions) == len(
            env.users
        ), "Number of actions must equal overall UEs."

        users = sorted(env.users, key=lambda x: x.ue_id)
        return {ue_id: action for ue_id, action in zip(users, actions)}

    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Select from & flatten observations from MA setting."""
        # select observations considered in the central setting
        obs = {
            ue_id: [obs_dict[key] for key in cls.features]
            for ue_id, obs_dict in env.features().items()
        }

        # flatten observation to single vector
        return np.concatenate([o for ue_obs in obs.values() for o in ue_obs])

    @classmethod
    def reward(cls, env):
        """The central agent receives the average UE utility as reward."""
        utilities = np.asarray([utility for utility in env.utilities.values()])
        # assert that rewards are in range [-1, +1]
        bounded = np.logical_and(utilities >= -1, utilities <= 1).all()
        assert bounded, "Utilities must be in range [-1, +1]"

        # return average utility of UEs to central agent as reward
        return np.mean(utilities)

    @classmethod
    def check(cls, env):
        assert [
            ue.stime <= 0.0 and ue.extime >= env.EP_MAX_TIME for ue in env.users
        ], "Central environment cannot handle a changing number of UEs."
