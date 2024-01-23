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

# Translating actions to coordinate changes
MOVES = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
NOOP = -1

# Object encodings
EMPTY = 0
WALL = 1
TARGET = 2
AGENT = 3
BOX = 4
TARGET_AGENT = 5
TARGET_BOX = 6

# Environment Variables
N_BOXES = 4
GRID_SIZE = 10

# Reward Function
LEVEL_COMPLETE_BONUS = 10
SINGLE_BOX_BONUS = 1
STEP_BONUS = -0.1
