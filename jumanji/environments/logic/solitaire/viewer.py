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

from typing import Optional, Sequence

import matplotlib
import matplotlib.animation

from jumanji.environments.logic.solitaire.types import State
from jumanji.viewer import Viewer


class SolitaireViewer(Viewer):
    def __init__(
        self,
        name: str = "Solitaire",
        board_size: int = 7,
    ) -> None:
        """Viewer for the Solitaire environment.

        Args:
            name: the window name to be used when initialising the window.
            board_size: size of the board.
        """
        self._name = name
        self._board_size = board_size

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(self, state: State) -> None:
        ...

    def close(self) -> None:
        ...

    def animate(
        self, states: Sequence, interval: int, save_path: Optional[str]
    ) -> matplotlib.animation.FuncAnimation:
        ...
