from typing import NamedTuple, Optional

import haiku as hk
import jax.numpy as jnp
import optax
from chex import ArrayTree


class Transition(NamedTuple):
    """Container for a transition."""

    observation: ArrayTree
    action: ArrayTree
    reward: ArrayTree
    discount: ArrayTree
    next_observation: ArrayTree


class TrainingState(NamedTuple):
    """Container for the variables used during the training of an agent."""

    params: Optional[hk.Params] = None
    opt_state: Optional[optax.OptState] = None
    counter: jnp.int32 = jnp.int32(0)
