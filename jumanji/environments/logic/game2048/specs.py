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
from jumanji.environments.logic.game2048.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        board_spec: specs.Array,
        action_mask_spec: specs.Array,
    ):
        name = (
            "Observation(\n"
            f"\tboard_spec: {board_spec.name},\n"
            f"\taction_mask_spec: {action_mask_spec.name},\n"
            ")"
        )
        super().__init__(name=name)
        self.board_spec = board_spec
        self.action_mask_spec = action_mask_spec

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tboard_spec={repr(self.board_spec)},\n"
            f"\taction_mask_spec={repr(self.action_mask_spec)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate values for the observations board and action_mask."""
        return Observation(
            board=self.board_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Game2048 Observation conforms to the spec.

        Args:
            value: a Game2048 Observation

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            board=self.board_spec.validate(value.board),
            action_mask=self.action_mask_spec.validate(value.action_mask),
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
            "board_spec": self.board_spec,
            "action_mask_spec": self.action_mask_spec,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
