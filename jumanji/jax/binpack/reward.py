import chex
import jax.numpy as jnp

from jumanji.jax.binpack.types import State
from jumanji.jax.types import Action


def sparse_linear_reward(state: State, action: Action, done: jnp.bool_) -> chex.Array:
    # TODO
    return jnp.zeros((), float)
