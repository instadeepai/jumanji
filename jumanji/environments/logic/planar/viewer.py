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

from typing import Callable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from jumanji.environments.logic.planar.types import State

# style definition
_STYLE = {
    "font_weight": "normal",
    "with_labels": True,
    "node_color": "#179E75",
    "font_color": "w",
    "node_size": 500,
    "font_size": 10,
    "node_shape": "o",
    "width": 1.25,
}

Viewer = Callable[[State], plt.Figure]


def _build_networkx_graph(state: State) -> nx.Graph:
    # create an empty graph
    graph = nx.Graph()

    # Add nodes
    num_nodes = int(state.nodes.shape[0])
    graph.add_nodes_from(range(num_nodes))

    # Add edges
    edges = np.asarray(state.edges).tolist()
    graph.add_edges_from(edges)

    return graph


def networkx_viewer(state: State) -> plt.Figure:
    fig, ax = plt.subplots()

    # Build and draw the graph
    pos = np.asarray(state.nodes)
    graph = _build_networkx_graph(state)
    nx.draw(graph, pos=pos, ax=ax, **_STYLE)

    return fig


def networkx_animation(states: Sequence[State]) -> matplotlib.animation.FuncAnimation:
    fig, ax = plt.subplots()

    # Build the graph (the graph topology remains fixed during an episode)
    graph = _build_networkx_graph(states[0])

    def _animate(state: State) -> None:
        # Clear the axes first
        ax.cla()

        # Draw the graph at the current nodes' positions.
        nx.draw(graph, pos=np.asarray(state.nodes), ax=ax, **_STYLE)

    return matplotlib.animation.FuncAnimation(
        fig=fig,
        func=_animate,
        frames=states,
    )
