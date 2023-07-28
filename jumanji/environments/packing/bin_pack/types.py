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

from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import chex
import jax.numpy as jnp
from typing_extensions import TypeAlias

from jumanji.environments.packing.bin_pack.space import Space

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


Container: TypeAlias = Space
EMS: TypeAlias = Space


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


class ValuedItem(NamedTuple):
    x_len: chex.Numeric
    y_len: chex.Numeric
    z_len: chex.Numeric
    value: chex.Numeric


def valued_item_from_space_and_max_value(
    space: Space, value: chex.Numeric
) -> ValuedItem:
    return ValuedItem(
        x_len=space.x2 - space.x1,
        y_len=space.y2 - space.y1,
        z_len=space.z2 - space.z1,
        value=value,
    )


ItemType: TypeAlias = Union[Item, ValuedItem]


def rotated_items_from_space(
    space: Space, value: Optional[chex.Numeric] = None
) -> ItemType:
    x_len = jnp.asarray(
        [
            # x along X, y along Y, z along Z (orientation A of DeepPack)
            space.x2 - space.x1,
            # x along X, z along Y, y along Z (Orientation B of DeepPack)
            space.x2 - space.x1,
            # z along X, y along Y, x along Z (Orientation C of DeepPack)
            space.z2 - space.z1,
            # y along X, x along Y, z along Z (Orientation D of DeepPack)
            space.y2 - space.y1,
            # z along X, x along Y, y along Z (Orientation E of DeepPack)
            space.z2 - space.z1,
            # y along X, z along Y, x along Z (Orientation F of deepPack)
            space.y2 - space.y1,
        ],
    )
    y_len = jnp.asarray(
        [
            space.y2 - space.y1,
            space.z2 - space.z1,
            space.y2 - space.y1,
            space.x2 - space.x1,
            space.x2 - space.x1,
            space.z2 - space.z1,
        ]
    )
    z_len = jnp.asarray(
        [
            space.z2 - space.z1,
            space.y2 - space.y1,
            space.x2 - space.x1,
            space.z2 - space.z1,
            space.y2 - space.y1,
            space.x2 - space.x1,
        ]
    )
    if value is None:
        return Item(
            x_len=x_len,
            y_len=y_len,
            z_len=z_len,
        )
    else:
        return ValuedItem(
            x_len=x_len, y_len=y_len, z_len=z_len, value=jnp.asarray(6 * [value])
        )


def item_fits_in_item(item: ItemType, other_item: ItemType) -> chex.Array:
    """Check if an item is smaller than another one."""
    return (
        (item.x_len <= other_item.x_len)
        & (item.y_len <= other_item.y_len)
        & (item.z_len <= other_item.z_len)
    )


def item_volume(item: ItemType) -> chex.Array:
    """Returns the volume as a float to prevent from overflow with 32 bits."""
    x_len = jnp.asarray(item.x_len, float)
    y_len = jnp.asarray(item.y_len, float)
    z_len = jnp.asarray(item.z_len, float)
    return x_len * y_len * z_len


def item_value(item: ItemType) -> chex.Array:
    if not isinstance(item, ValuedItem):
        raise ValueError(f"Trying to obtain the value of an item of type {type(item)}")
    return jnp.asarray(item.value, float)


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


def space_from_item_and_location(item: ItemType, location: Location) -> Space:
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
    """
    container: space defined by 2 points, i.e. 6 coordinates.
    ems: empty maximal spaces (EMSs) in the container, each defined by 2 points (6 coordinates).
    ems_mask: array of booleans that indicate the EMSs that are valid.
    items: defined by 3 attributes (x, y, z).
    items_mask: array of booleans that indicate the items that can be packed.
    items_placed: array of booleans that indicate the items that have been placed so far.
    items_location: locations of items in the container, defined by 3 coordinates (x, y, x).
    action_mask: array of booleans that indicate the valid actions, i.e. EMSs and items that can
        be chosen.
    sorted_ems_indexes: EMS indexes that are sorted by decreasing volume order.
    key: random key used for auto-reset.
    """

    container: Container  # leaves of shape ()
    ems: EMS  # leaves of shape (max_num_ems,)
    ems_mask: chex.Array  # (max_num_ems,)
    # Since the items are allowed to take one of 6 orientations the items, items_mask , items_placed
    # and action_mask tensors all have an extra dimension of size 6 representing the orientation
    # that the items takes. The agent however sees the several orientations of each items as 6
    # different items among which it can only pack one.
    items: ItemType  # leaves of shape (6,max_num_items,)
    items_mask: chex.Array  # (6,max_num_items)
    items_placed: chex.Array  # (6,max_num_items)
    items_location: Location  # leaves of shape (max_num_items,)
    action_mask: Optional[chex.Array]  # (6, obs_num_ems, max_num_items)
    sorted_ems_indexes: chex.Array  # (max_num_ems,)
    instance_max_item_value_magnitude: chex.Numeric  # () - only for value based optimisation
    instance_total_value: chex.Numeric  # leaves of shape () - only for value based optimisation
    key: chex.PRNGKey  # (2,)
    nb_items: int  # leaves of shape ()


class Observation(NamedTuple):
    """
    ems: empty maximal spaces (EMSs) in the container, defined by 2 points (6 coordinates).
    ems_mask: array of booleans that indicate the EMSs that are valid.
    items: defined by 3 attributes (x, y, z).
    items_mask: array of booleans that indicate the items that are valid.
    items_placed: array of booleans that indicate the items that have been placed so far.
    action_mask: array of booleans that indicate the feasible actions, i.e. EMSs and items that can
        be chosen.
    """

    ems: EMS  # leaves of shape (obs_num_ems,)
    ems_mask: chex.Array  # (obs_num_ems,)
    items: ItemType  # leaves of shape (max_num_items,)
    items_mask: chex.Array  # (6*max_num_items,)
    items_placed: chex.Array  # (6*max_num_items,)
    action_mask: chex.Array  # (obs_num_ems, 6*max_num_items)
