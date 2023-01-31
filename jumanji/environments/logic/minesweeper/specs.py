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
from jumanji.environments.logic.minesweeper.types import Observation


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        board: specs.BoundedArray,
        action_mask: specs.BoundedArray,
        num_mines: specs.BoundedArray,
        step_count: specs.BoundedArray,
    ):
        name = (
            "Observation(\n"
            f"\tboard: {board.name},\n"
            f"\taction_mask: {action_mask.name},\n"
            f"\tnum_mines: {num_mines.name},\n"
            f"\tstep_count: {step_count.name},\n"
            ")"
        )
        super().__init__(name=name)
        self.board = board
        self.action_mask = action_mask
        self.num_mines = num_mines
        self.step_count = step_count

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tboard={repr(self.board)},\n"
            f"\taction_mask={repr(self.action_mask)},\n"
            f"\tnum_mines={repr(self.num_mines)},\n"
            f"\tstep_count={repr(self.step_count)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate values for the board observation."""
        return Observation(
            board=self.board.generate_value(),
            action_mask=self.action_mask.generate_value(),
            num_mines=self.num_mines.generate_value(),
            step_count=self.step_count.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a Minesweeper Observation conforms to the spec.

        Args:
            value: a Minesweeper Observation

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            board=self.board.validate(value.board),
            action_mask=self.action_mask.validate(value.action_mask),
            num_mines=self.num_mines.validate(value.num_mines),
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
            "board": self.board,
            "action_mask": self.action_mask,
            "num_mines": self.num_mines,
            "step_count": self.step_count,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
