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

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

from enum import IntEnum

import chex

# Grid channels
_SHELVES = 0
_AGENTS = 1


class Action(IntEnum):
    """An enumeration of possible actions
    that an agent can take in the warehouse.

    NOOP - represents no operation.
    FORWARD - move forward.
    LEFT - turn left.
    RIGHT - turn right.
    TOGGLE_LOAD - toggle loading/offloading a shelf.
    """

    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(IntEnum):
    """An enumeration of possible directions
    that an agent can take in the warehouse.

    UP - move up.
    DOWN - move down.
    LEFT - move left.
    RIGHT - move right.
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Position(NamedTuple):
    """A class to represent the 2D coordinate position of entities

    x : the x-position of the entity.
    y : the y-position of the entity.
    """

    x: chex.Array
    y: chex.Array


class Agent(NamedTuple):
    """A class to represent an Agent in the warehouse

    position : the (x,y) position of the agent.
    direction : the direction the agent is facing.
    is_carrying : whether the agent is carrying a shelf or not.
    """

    position: Position
    direction: chex.Array
    is_carrying: chex.Array


class Shelf(NamedTuple):
    """A class to represent a Shelf in the warehouse.

    position : the (x,y) position of the shelf.
    is_requested : whether the shelf is requested for delivery.
    """

    position: Position
    is_requested: chex.Array


@dataclass
class State:
    """A dataclass representing the state of the simulated warehouse.

    grid : an array representing the warehouse floor as a 2D grid with two separate channels
        one for the agents, and one for the shelves.
    agents : a pytree of Agent type with per agent leaves: [position, direction, is_carrying]
    shelves : a pytree of Shelf type with per shelf leaves: [position, is_requested]
    request_queue : the queue of requested shelves (by ID).
    step : an integer representing the current step of the episode.
    key : a pseudorandom number generator key.
    """

    grid: chex.Array  # (2, grid_width, grid_height)
    agents: Agent  # (num_agents,)
    shelves: Shelf  # (num_shelves,)
    request_queue: chex.Array  # (num_requested,)
    step: chex.Array
    key: chex.PRNGKey
