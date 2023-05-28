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


# Number of possible rotations (0 degrees, 90 degrees, 180 degrees and 270 degrees)
NUM_ROTATIONS = 4
# reward is calculated using the number of lines
# REWARD_LIST(nbr_full_lines)
REWARD_LIST = (0, 40, 100, 300, 1200)

# TETROMINOES_LIST is a collection of 7 distinct tetromino shapes,
# with each shape represented as a 4x4 grid of cells.
# Each tetromino has 4 possible rotations (0, 90, 180, and 270 degrees).
# `TETROMINOES_LIST` has shape (7, 4) where 7 is the number of tetromino types
# and 4 is the number of possible rotations of each tetromino.
TETROMINOES_LIST = (
    (
        (
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0),
        ),
        (
            (1, 1, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0),
        ),
        (
            (1, 1, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
    (
        (
            (0, 1, 1, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (0, 1, 1, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
    (
        (
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
    (
        (
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
    (
        (
            (1, 1, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (0, 1, 0, 0),
            (1, 1, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
    (
        (
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 1, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (0, 0, 1, 0),
            (1, 1, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
    (
        (
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 0, 0, 0),
            (1, 1, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 0, 0),
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        (
            (1, 1, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
)
