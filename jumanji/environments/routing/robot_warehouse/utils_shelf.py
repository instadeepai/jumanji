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

from typing import Tuple, Union

import chex

from jumanji.environments.routing.robot_warehouse.constants import _SHELVES
from jumanji.environments.routing.robot_warehouse.types import Position, Shelf
from jumanji.tree_utils import tree_add_element, tree_slice


def update_shelf(
    shelves: Shelf,
    shelf_id: chex.Array,
    attr: str,
    value: Union[chex.Array, Position],
) -> Shelf:
    """Update the attribute information of a specific shelf.

    Args:
        shelves: a pytree of Shelf type containing shelf information.
        shelf_id: unique ID identifying a specific shelf.
        attr: the attribute to update, e.g. `direction`, or `is_requested`.
        value: the new value to which the attribute is to be set.

    Returns:
        the shelf with the specified attribute updated to the given value.
    """
    params = {attr: value}
    shelf = tree_slice(shelves, shelf_id)
    shelf = shelf._replace(**params)
    shelves: Shelf = tree_add_element(shelves, shelf_id, shelf)
    return shelves


def set_new_shelf_position_if_carrying(
    grid: chex.Array,
    shelves: Shelf,
    cur_pos: chex.Array,
    new_pos: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Set the new position of the shelf if the agent is carrying one.

    Args:
        grid: the warehouse floor grid array.
        shelves: a pytree of Shelf type containing shelf information.
        cur_pos: the current position of the shelf.
        new_pos: the new position of the shelf.

    Returns:
        updated grid array and shelves pytree.
    """
    # update shelf position
    shelf_id = grid[_SHELVES, cur_pos.x, cur_pos.y]
    shelves = update_shelf(shelves, shelf_id - 1, "position", new_pos)

    # update shelf grid placement
    grid = grid.at[_SHELVES, cur_pos.x, cur_pos.y].set(0)
    grid = grid.at[_SHELVES, new_pos.x, new_pos.y].set(shelf_id)
    return grid, shelves
