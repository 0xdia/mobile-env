import itertools

import gymnasium as gym

from mobile_env.handlers.central import MComCentralHandler
from mobile_env.handlers.multi_agent import MComMAHandler
from mobile_env.scenarios.verylarge import MComVeryLarge

scenarios = {
    "verylarge": MComVeryLarge,
}
handlers = {"ma": MComMAHandler, "central": MComCentralHandler}

for scenario, handler in itertools.product(scenarios, handlers):
    env_name = scenarios[scenario].__name__
    config = {"handler": handlers[handler]}
    gym.envs.register(
        id=f"mobile-{scenario}-{handler}-v0",
        entry_point=f"mobile_env.scenarios.{scenario}:{env_name}",
        kwargs={"config": config},
    )
