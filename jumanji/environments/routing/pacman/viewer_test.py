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

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import py
import pytest

from jumanji.environments.pacman.env import PacMan
from jumanji.environments.pacman.viewer import PacManViewer


@pytest.fixture
def pac_man() -> PacMan:
    """Instantiates a default Game2048 environment."""
    return PacMan()


def test_pacman_viewer__render(
    pac_man: PacMan, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plt, "show", lambda fig: None)
    key = jax.random.PRNGKey(0)
    state, _ = pac_man.reset(key)
    viewer = PacManViewer("Pacman", render_mode="human")
    viewer.render(state)
    viewer.close()


def test_robot_warehouse_viewer__animate(pac_man: PacMan) -> None:
    key = jax.random.PRNGKey(0)
    state, _ = jax.jit(pac_man.reset)(key)

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, jnp.arange(4), shape=(1,))[0]
        state, _ = jax.jit(pac_man.step)(state, action)
        states.append(state)

    viewer = PacManViewer("Pacman", render_mode="human")
    viewer.animate(states)
    viewer.close()


def test_robot_warehouse_viewer__save_animation(
    pac_man: PacMan, tmpdir: py.path.local
) -> None:
    key = jax.random.PRNGKey(0)
    state, _ = jax.jit(pac_man.reset)(key)

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, jnp.arange(4), shape=(1,))[0]
        state, _ = jax.jit(pac_man.step)(state, action)
        states.append(state)

    viewer = PacManViewer("Pacman", render_mode="human")
    animation = viewer.animate(states)
    assert isinstance(animation, matplotlib.animation.Animation)

    save_path = str(tmpdir.join("/pacman_animation_test.gif"))
    animation.save(save_path)
    viewer.close()
