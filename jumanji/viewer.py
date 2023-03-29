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

"""Abstract environment viewer class."""

import abc
from typing import Generic, Optional, Sequence

import matplotlib
from numpy.typing import NDArray

from jumanji.env import State


class Viewer(abc.ABC, Generic[State]):
    """Abstract viewer class to support rendering and animation. This interface assumes
    that matplotlib is used for rendering the environment in question.
    """

    @abc.abstractmethod
    def render(self, state: State) -> Optional[NDArray]:
        """Render frames of the environment for a given state using matplotlib.

        Args:
            state: `State` object corresponding to the new state of the environment.
        """

    @abc.abstractmethod
    def animate(
        self,
        states: Sequence[State],
        interval: int,
        save_path: Optional[str],
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Perform any necessary cleanup. Environments will automatically :meth:`close()`
        themselves when garbage collected or when the program exits.
        """
