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

import chex
import jax
import matplotlib
import matplotlib.pyplot as plt
import pytest

from jumanji.environments.routing.lbf import LevelBasedForaging
from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer


def test_lbf_viewer_render(
    lbf_environment: LevelBasedForaging,
    monkeypatch: pytest.MonkeyPatch,
    key: chex.PRNGKey,
) -> None:
    """Test rendering using LevelBasedForagingViewer."""

    monkeypatch.setattr(plt, "show", lambda fig: None)
    state, _ = lbf_environment.reset(key)
    grid_size = lbf_environment._generator.grid_size

    viewer = LevelBasedForagingViewer(grid_size)
    viewer.render(state)
    viewer.close()


def test_lbf_viewer_animate(
    lbf_environment: LevelBasedForaging, key: chex.PRNGKey
) -> None:
    """Test animation using LevelBasedForagingViewer."""

    state, _ = jax.jit(lbf_environment.reset)(key)
    grid_size = lbf_environment._generator.grid_size

    states = [state]
    for _ in range(5):
        key, _ = jax.random.split(key)
        action = lbf_environment.action_spec().generate_value()
        state, _ = jax.jit(lbf_environment.step)(state, action)
        states.append(state)

    viewer = LevelBasedForagingViewer(grid_size)
    viewer.animate(states)
    viewer.close()


def test_lbf_viewer_save_animation(
    lbf_environment: LevelBasedForaging, tmp_path: str, key: chex.PRNGKey
) -> None:
    """Test saving animation using LevelBasedForagingViewer."""

    state, _ = jax.jit(lbf_environment.reset)(key)
    grid_size = lbf_environment._generator.grid_size

    states = [state]
    for _ in range(5):
        key, _ = jax.random.split(key)
        action = lbf_environment.action_spec().generate_value()
        state, _ = jax.jit(lbf_environment.step)(state, action)
        states.append(state)

    viewer = LevelBasedForagingViewer(grid_size)
    animation = viewer.animate(states)
    assert isinstance(animation, matplotlib.animation.Animation)

    # Use tmp_path directly to create the file path
    save_path = f"{tmp_path}/lbf_animation_test.gif"
    animation.save(save_path)
    viewer.close()
