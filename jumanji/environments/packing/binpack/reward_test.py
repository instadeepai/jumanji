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

from jumanji.environments.packing.binpack.env import BinPack
from jumanji.environments.packing.binpack.reward import sparse_linear_reward
from jumanji.environments.packing.binpack.types import State


def test__sparse_linear_reward(binpack_env: BinPack, dummy_instance: State) -> None:
    # Checks that reward is sparse, i.e. 0 if not done.
    assert sparse_linear_reward(dummy_instance, dummy_instance, None, done=False) == 0.0

    # If ends at the beginning of the episode, the reward should be also 0.
    assert sparse_linear_reward(dummy_instance, dummy_instance, None, done=True) == 0.0

    # After placing one item and ending the episode, the reward should be > 0.
    dummy_instance.items_placed = dummy_instance.items_placed.at[0].set(True)
    assert sparse_linear_reward(dummy_instance, dummy_instance, None, done=True) > 0.0

    # After taking an action, the reward becomes > 0 if end of the episode, else 0.
    next_state, _ = binpack_env.step(dummy_instance, jnp.array([0, 0]))
    assert sparse_linear_reward(dummy_instance, next_state, None, done=True) > 0.0
    assert sparse_linear_reward(dummy_instance, next_state, None, done=False) == 0.0
