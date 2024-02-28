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

x = jnp.array([[1, 4], [1, 2], [2, 5]])
y = jnp.array([[1, 3], [1, 3], [2, 4]])

what, indices, counts = jnp.unique(
    y, return_inverse=True, return_counts=True, size=len(y), axis=0
)
duplicates = ~(counts[indices] == 1)


# duplicates = jnp.broadcast_to(duplicates[:, None], (2,2))

# If there are duplicates, use the original agent position.
new_positions = jnp.where(
    duplicates,
    x,  # original agent position
    y,  # moved agent position
)
x = 1
# rez==1
# Array([ True,  True], dtype=bool)
# (rez==1)[:, None]
# Array([[ True],
#        [ True]], dtype=bool)
# (rez==1)[None, :]
# Array([[ True,  True]], dtype=bool)
# (rez==1)[:, None]
# Array([[ True],
#        [ True]], dtype=bool)
# (rez==1)[:, None].shape
# (2, 1)
# jnp.broadcast_to((rez==1)[:, None], (2,2))
# Array([[ True,  True],
#        [ True,  True]], dtype=bool)
