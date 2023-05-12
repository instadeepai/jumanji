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

import jax.numpy as jnp
import jax.random as random
import matplotlib.animation
import matplotlib.pyplot as plt
import pytest

from jumanji.environments.logic.graph_coloring import GraphColoring


def test_render(monkeypatch: pytest.MonkeyPatch, graph_coloring: GraphColoring) -> None:
    """Check that the render method builds the figure but does not display it."""
    monkeypatch.setattr(plt, "show", lambda fig: None)
    key = random.PRNGKey(0)
    state, _ = graph_coloring.reset(key)

    graph_coloring.render(state)
    graph_coloring.close()


def test_animate(graph_coloring: GraphColoring) -> None:
    """Check that the animation method creates the animation correctly and can save to a gif."""
    key = random.PRNGKey(0)
    state, _ = graph_coloring.reset(key)

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        action = jnp.array(0)
        new_state, _ = graph_coloring.step(state, action)
        states.append(new_state)
        state = new_state

    animation = graph_coloring.animate(states, interval=500)
    assert isinstance(animation, matplotlib.animation.Animation)


def test_save_animation(graph_coloring: GraphColoring) -> None:

    key = random.PRNGKey(0)
    state, _ = graph_coloring.reset(key)

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        action = jnp.array(0)
        new_state, _ = graph_coloring.step(state, action)
        states.append(new_state)
        state = new_state

    save_path = "animation_test.gif"
    graph_coloring.animate(states, interval=500, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)
