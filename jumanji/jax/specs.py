from typing import NamedTuple

import dm_env.specs

from jumanji.jax.env import JaxEnv


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""

    observations: dm_env.specs.Array
    actions: dm_env.specs.Array
    rewards: dm_env.specs.Array
    discounts: dm_env.specs.Array


def make_environment_spec(jax_env: JaxEnv) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=jax_env.observation_spec(),
        actions=jax_env.action_spec(),
        rewards=jax_env.reward_spec(),
        discounts=jax_env.discount_spec(),
    )
