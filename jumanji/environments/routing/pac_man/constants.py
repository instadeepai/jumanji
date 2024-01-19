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

import jax.numpy as jnp

MOVES = jnp.array(
    [[0, -1], [-1, 0], [0, 1], [1, 0], [0, 0]]
)  # Up, Right, Down, Left, No-op


# Default Maze design
DEFAULT_MAZE = [
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    "X  S         XX         S  X",
    "X XXXX XXXXX XX XXXXX XXXX X",
    "X XXXXOXXXXX XX XXXXXOXXXX X",
    "X XXXX XXXXX XX XXXXX XXXX X",
    "X                          X",
    "X XXXX XX XXXXXXXX XX XXXX X",
    "X XXXX XX XXXXXXXX XX XXXX X",
    "X      XX   TXXT   XX      X",
    "XXXXXX XXXXX XX XXXXX XXXXXX",
    "XXXXXX XXXXX XX XXXXX XXXXXX",
    "XXXXXX XXT        TXX XXXXXX",
    "XXXXXX XX XXX XXXX XX XXXXXX",
    "XXXXXX XX X  G   X XX XXXXXX",
    "           GXXXXG           ",
    "XXXXXX XX X  G   X XX XXXXXX",
    "XXXXXX XX XXX XXXX XX XXXXXX",
    "XXXXXX XX          XX XXXXXX",
    "XXXXXX XX XXXXXXXX XX XXXXXX",
    "XXXXXX XX XXXXXXXX XX XXXXXX",
    "X            XX            X",
    "X XXXX XXXXX XX XXXXX XXXX X",
    "X XXXX XXXXX XX XXXXX XXXX X",
    "X   XX S     P     S  XX   X",
    "XXX XX XX XXXXXXXX XX XX XXX",
    "XXX XX XX XXXXXXXX XX XX XXX",
    "X      XX    XX    XX      X",
    "X XXXXXXXXXX XX XXXXXXXXXX X",
    "X XXXXXXXXXX XX XXXXXXXXXX X",
    "X       O             O    X",
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
]
