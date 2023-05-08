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
import matplotlib.pyplot as plt

from jumanji.environments.logic.graph_coloring import GraphColoring
from jumanji.environments.logic.graph_coloring.viewer import GraphColoringViewer


def test_render(graph_coloring: GraphColoring) -> None:
    key = random.PRNGKey(0)
    state, _ = graph_coloring.reset(key)
    num_nodes, _ = graph_coloring.generator.specs()

    viewer = GraphColoringViewer(num_nodes)
    viewer.render(state)

    plt.show()


def test_animate(graph_coloring: GraphColoring) -> None:

    key = random.PRNGKey(0)
    state, _ = graph_coloring.reset(key)
    num_nodes, _ = graph_coloring.generator.specs()

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        action = jnp.array(0)
        new_state, _ = graph_coloring.step(state, action)
        states.append(new_state)
        state = new_state

    viewer = GraphColoringViewer(num_nodes)
    viewer.animate(states, interval=500)
    plt.show()


def test_save_render(graph_coloring: GraphColoring) -> None:

    key = random.PRNGKey(0)
    state, _ = graph_coloring.reset(key)
    num_nodes, _ = graph_coloring.generator.specs()

    viewer = GraphColoringViewer(num_nodes)

    save_path = "render_test.png"
    viewer.render(state, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)


def test_save_animation(graph_coloring: GraphColoring) -> None:

    key = random.PRNGKey(0)
    state, _ = graph_coloring.reset(key)
    num_nodes, _ = graph_coloring.generator.specs()

    num_steps = 5
    states = [state]
    for _ in range(num_steps - 1):
        action = jnp.array(0)
        new_state, _ = graph_coloring.step(state, action)
        states.append(new_state)
        state = new_state

    viewer = GraphColoringViewer(num_nodes)

    save_path = "animation_test.gif"
    viewer.animate(states, interval=500, save_path=save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)
