from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env, StarCraft2SortEnv, StarCraft2Not0Env, StarCraft2Set1Env
import sys
import os



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2_sort"] = partial(env_fn, env=StarCraft2SortEnv)
REGISTRY["sc2_not_0"] = partial(env_fn, env=StarCraft2Not0Env)
REGISTRY["sc2_set_1"] = partial(env_fn, env=StarCraft2Set1Env)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

if __name__ == "__main__":
    env = REGISTRY['sc2']()