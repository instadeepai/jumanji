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

from jumanji.environments.games.snake.types import Position


def test_position__eq() -> None:
    """Validates the equality check of the Position type.
    Checks whether some positions are equal or not.
    """
    pos1 = Position(3, 5)
    pos2 = Position(3, 5)
    assert pos1 == pos2
    pos3 = Position(4, 5)
    assert pos3 != pos1
