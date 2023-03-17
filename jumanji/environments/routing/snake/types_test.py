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

from jumanji.environments.routing.snake.types import Position


def test_position__eq() -> None:
    """Validates the equality check of the Position type.
    Checks whether some positions are equal or not.
    """
    position = Position(3, 5)
    assert position == Position(3, 5)
    assert position != Position(4, 5)
    assert position != Position(3, 4)


def test_position__add() -> None:
    """Validates the addition of two `Position` instances."""
    assert Position(3, 5) + Position(3, 5) == Position(6, 10)
    assert Position(0, 1) + Position(2, 3) == Position(2, 4)
    assert Position(-2, 1) + Position(1, -4) != Position(0, 0)
