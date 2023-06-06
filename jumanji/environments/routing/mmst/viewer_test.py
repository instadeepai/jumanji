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

import os
from typing import Tuple

from jumanji.environments.routing.mmst.env import MMST
from jumanji.environments.routing.mmst.types import State
from jumanji.environments.routing.mmst.viewer import MMSTViewer
from jumanji.types import TimeStep


def test__render(deterministic_mmst_env: Tuple[MMST, State, TimeStep]) -> None:
    """Test that the viewer works and the frame is saved."""

    env, state, _ = deterministic_mmst_env
    viewer = MMSTViewer(
        env.num_agents,
    )

    viewer.render(state)


def test__animation(deterministic_mmst_env: Tuple[MMST, State, TimeStep]) -> None:
    """Test the viewer's animation function."""

    env, state, _ = deterministic_mmst_env
    viewer = MMSTViewer(
        env.num_agents,
    )
    save_path = "render_test_gif.gif"

    states = 2 * [state]
    viewer.animate(states, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)
