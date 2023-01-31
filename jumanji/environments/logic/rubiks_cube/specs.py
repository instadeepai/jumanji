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

from jumanji import specs
from jumanji.environments.logic.rubiks_cube.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        cube: specs.BoundedArray,
        step_count: specs.BoundedArray,
    ):
        name = (
            "Observation(\n"
            f"\tcube: {cube.name},\n"
            f"\tstep_count: {step_count.name},\n"
            ")"
        )
        super().__init__(name=name)
        self.cube = cube
        self.step_count = step_count

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tcube={repr(self.cube)},\n"
            f"\tstep_count={repr(self.step_count)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate values for the cube observation."""
        return Observation(
            cube=self.cube.generate_value(),
            step_count=self.step_count.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a RubiksCube Observation conforms to the spec.

        Args:
            value: a RubiksCube Observation

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            cube=self.cube.validate(value.cube),
            step_count=self.step_count.validate(value.step_count),
        )
        return observation

    def replace(self, **kwargs: Any) -> "ObservationSpec":
        """Returns a new copy of `ObservationSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ObservationSpec`.
        """
        all_kwargs = {
            "cube": self.cube,
            "step_count": self.step_count,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
