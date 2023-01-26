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

from typing import Any, Callable, NamedTuple

import chex
import haiku as hk


class FeedForwardNetwork(NamedTuple):
    """Networks are meant to take a batch of observations: shape (B, ...)."""

    init: Callable[[chex.PRNGKey, Any], hk.Params]
    apply: Callable[[hk.Params, Any], chex.Array]
