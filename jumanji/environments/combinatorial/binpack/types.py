# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax.numpy as jnp

from jumanji.environments.combinatorial.binpack.space import Space

Container = Space
EMS = Space


def empty_ems() -> EMS:
    """Returns an empty EMS located at (0, 0, 0)."""
    return EMS(x1=0, x2=0, y1=0, y2=0, z1=0, z2=0).astype(jnp.int32)


class Item(NamedTuple):
    x_len: chex.Numeric
    y_len: chex.Numeric
    z_len: chex.Numeric


def item_from_space(space: Space) -> Item:
    """Convert a space to an item whose length on each dimension is the length of the space."""
    return Item(
        x_len=space.x2 - space.x1,
        y_len=space.y2 - space.y1,
        z_len=space.z2 - space.z1,
    )


def item_fits_in_item(item: Item, other_item: Item) -> jnp.bool_:
    """Check if an item is smaller than another one."""
    return (
        (item.x_len <= other_item.x_len)
        & (item.y_len <= other_item.y_len)
        & (item.z_len <= other_item.z_len)
    )


def item_volume(item: Item) -> jnp.float_:
    """Returns the volume as a float to prevent from overflow with 32 bits."""
    x_len = jnp.float_(item.x_len)
    y_len = jnp.float_(item.y_len)
    z_len = jnp.float_(item.z_len)
    return x_len * y_len * z_len


class Location(NamedTuple):
    x: chex.Numeric
    y: chex.Numeric
    z: chex.Numeric


def location_from_space(space: Space) -> Location:
    """Returns the location of a space, i.e. the coordinates of its bottom left corner.

    Args:
        space: space object from which to get the location.

    Returns:
        location of the space object (x1, y1, z1).

    """
    return Location(
        x=space.x1,
        y=space.y1,
        z=space.z1,
    )


def space_from_item_and_location(item: Item, location: Location) -> Space:
    """Returns a space from an item at a particular location. The bottom left corner is given
    by the location while the top right is the location plus the item dimensions.
    """
    return Space(
        x1=location.x,
        x2=location.x + item.x_len,
        y1=location.y,
        y2=location.y + item.y_len,
        z1=location.z,
        z2=location.z + item.z_len,
    )


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
    key: chex.Array  # Source of randomness for instances with a random reset.

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
