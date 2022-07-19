from typing import TYPE_CHECKING, Callable, NamedTuple, Optional

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp

from jumanji.jax.binpack.space import Space
from jumanji.jax.types import Action

Container = Space
EMS = Space


class Item(NamedTuple):
    x_len: chex.Numeric
    y_len: chex.Numeric
    z_len: chex.Numeric


class Location(NamedTuple):
    x: chex.Numeric
    y: chex.Numeric
    z: chex.Numeric


@dataclass
class State:
    container: Container  # Dimension of the container for the current episode | shape (,)
    ems: EMS  # Leaves (x1, x2, y1, y2, z1, z2) | shape (max_num_ems,)
    ems_mask: chex.Array  # True if the ems exists | shape (max_num_ems,)
    items: Item  # Leaves (x_len, y_len, z_len) | shape (max_num_items,)
    items_mask: chex.Array  # True if items exist | shape (max_num_items,)
    items_placed: chex.Array  # True if items are placed in the container | shape (max_num_items,)
    items_location: Location  # Leaves (x, y, z) | shape (max_num_items,)
    action_mask: Optional[chex.Array]  # Array | shape (obs_num_ems, max_num_items)
    sorted_ems_indexes: chex.Array  # Array | shape (max_num_ems,)

    def __repr__(self) -> str:
        class_name = f"{self.__class__.__name__}("
        attribute_values = [
            f"\n\t{key}={repr(value)}," for key, value in self.__dict__.items()
        ]
        end_str = "\n)"
        return "".join(
            [
                class_name,
                *attribute_values,
                end_str,
            ]
        )


class Observation(NamedTuple):
    ems: EMS  # Leaves (x1, x2, y1, y2, z1, z2) | shape (obs_num_ems,)
    ems_mask: chex.Array  # True if ems exist | shape (obs_num_ems,)
    items: Item  # Leaves (x_len, y_len, z_len) | shape (max_num_items,)
    items_mask: chex.Array  # True if items exist | shape (max_num_items,)
    items_placed: chex.Array  # True if items are placed in the container | shape (max_num_items,)
    action_mask: chex.Array  # Joint action mask | shape (obs_num_ems, max_num_items)


RewardFn = Callable[[State, Action, jnp.bool_], chex.Array]
