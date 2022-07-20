import chex
import jax.numpy as jnp

from jumanji.binpack.types import State
from jumanji.types import Action


def sparse_linear_reward(state: State, action: Action, done: jnp.bool_) -> chex.Array:
    # TODO
    return jnp.zeros((), float)
