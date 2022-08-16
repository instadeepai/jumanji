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

from typing import TYPE_CHECKING, Any

import chex
import jax.numpy as jnp

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Space:
    x1: chex.Numeric
    x2: chex.Numeric
    y1: chex.Numeric
    y2: chex.Numeric
    z1: chex.Numeric
    z2: chex.Numeric

    def astype(self, dtype: Any) -> "Space":
        space_dict = {
            key: jnp.asarray(value, dtype) for key, value in self.__dict__.items()
        }
        return Space(**space_dict)

    def get_axis_value(self, axis: str, index: int) -> chex.Numeric:
        """Dynamically returns the correct attribute given the axis ("x", "y" or "z") and the index
        (1 or 2).

        Args:
            axis: string that can either be "x", "y" or "z".
            index: index of attribute to get, either 1 or 2.

        Returns:
            attribute of space matching the axis and index value.

        Example:
            ```python
            space.y2 is space.get_axis_value("y", 2)  # True
            ```
        """
        return getattr(self, f"{axis}{index}")

    def set_axis_value(self, axis: str, index: int, value: chex.Numeric) -> None:
        """Dynamically sets the correct attribute given the axis ("x", "y" or "z"), the index
        (1 or 2) and the value to set.

        Args:
            axis: string that can either be "x", "y" or "z".
            index: index of attribute to get, either 1 or 2.
            value: value to set the attribute to.

        Example:
            ```python
            space.y2 = 10
            # Using the setter gives:
            space.set_axis_value("y", 2, 10)
            ```
        """
        return setattr(self, f"{axis}{index}", value)

    def __repr__(self) -> str:
        return (
            "Space(\n"
            f"\tx1={repr(self.x1)}, x2={repr(self.x2)},\n"
            f"\ty1={repr(self.y1)}, y2={repr(self.y2)},\n"
            f"\tz1={repr(self.z1)}, z2={repr(self.z2)},\n"
            ")"
        )
