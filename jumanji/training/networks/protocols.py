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

from typing import Any

import chex
from typing_extensions import Protocol


class RandomPolicy(Protocol):
    """Callable specification for the random policy."""

    def __call__(
        self,
        observation: Any,
        key: chex.PRNGKey,
    ) -> chex.Array:
        """A random policy given an environment-specific observation.

        Args:
            observation: environment observation.
            key: random key for action selection.

        Returns:
            action
        """


class ObservationWithActionMaskProtocol(Protocol):
    """Make sure that the Observation has a field called action_mask."""

    action_mask: chex.Array
