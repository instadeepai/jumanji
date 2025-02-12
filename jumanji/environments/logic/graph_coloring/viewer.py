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

from itertools import pairwise
from typing import List, Optional, Sequence, Tuple

import chex
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist

import jumanji.environments
from jumanji.environments.logic.graph_coloring.types import State
from jumanji.viewer import Viewer


class GraphColoringViewer(Viewer):
    FIGURE_SIZE = (10.0, 10.0)

    def __init__(
        self,
        name: str = "GraphColoring",
    ) -> None:
        self._name = name
        self._animation: Optional[animation.Animation] = None

    def render(
        self,
        state: State,
        save_path: Optional[str] = None,
    ) -> None:
        self._clear_display()
        self._set_params(state)
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax, state)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.2)

        self._display_human(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 500,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        self._set_params(states[0])
        fig = plt.figure(f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax = fig.add_subplot(111)
        plt.close(fig)
        nodes, labels, edges = self._prepare_figure(ax, states[0])

        def make_frame(state_pair: Tuple[State, State]) -> List[Artist]:
            prev_state, state = state_pair

            for circle, color in zip(nodes, state.colors, strict=False):
                circle.set(color=self._color_mapping[color])
            # Update node and edges if new episode
            if not np.array_equal(prev_state.adj_matrix, state.adj_matrix):
                pos = self._spring_layout(state.adj_matrix, self.num_nodes)
                for circle, label, xy in zip(nodes, labels, pos, strict=False):
                    circle.set_center(xy)
                    label.set(x=xy[0], y=xy[1])
                n = 0
                for i in range(self.num_nodes):
                    for j in range(i + 1, self.num_nodes):
                        edges[n].set(
                            xdata=[pos[i][0], pos[j][0]],
                            ydata=[pos[i][1], pos[j][1]],
                            visible=state.adj_matrix[i, j],
                        )
                        n += 1

                return nodes + edges

            else:
                return nodes

        _animation = animation.FuncAnimation(
            fig,
            make_frame,
            frames=pairwise(states),
            interval=interval,
            blit=False,
        )

        if save_path:
            _animation.save(save_path)

        return _animation

    def _set_params(self, state: State) -> None:
        self.num_nodes = state.adj_matrix.shape[0]
        self.node_scale = self._calculate_node_scale(self.num_nodes)
        self._color_mapping = self._create_color_mapping(self.num_nodes)

    def _prepare_figure(
        self, ax: plt.Axes, state: State
    ) -> Tuple[List[plt.Circle], List[plt.Text], List[plt.Line2D]]:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        pos = self._spring_layout(state.adj_matrix, self.num_nodes)
        edges = self._render_edges(ax, pos, state.adj_matrix, self.num_nodes)
        nodes, labels = self._render_nodes(ax, pos, state.colors)
        return nodes, labels, edges

    def close(self) -> None:
        plt.close(self._name)

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self._name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

    def _compute_repulsive_forces(
        self, repulsive_forces: np.ndarray, pos: np.ndarray, k: float, num_nodes: int
    ) -> np.ndarray:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                delta = pos[i] - pos[j]
                distance = np.linalg.norm(delta)
                direction = delta / (distance + 1e-6)
                force = k * k / (distance + 1e-6)
                repulsive_forces[i] += direction * force
                repulsive_forces[j] -= direction * force

        return repulsive_forces

    def _compute_attractive_forces(
        self,
        graph: chex.Array,
        attractive_forces: np.ndarray,
        pos: np.ndarray,
        k: float,
        num_nodes: int,
    ) -> np.ndarray:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if graph[i, j]:
                    delta = pos[i] - pos[j]
                    distance = np.linalg.norm(delta)
                    direction = delta / (distance + 1e-6)
                    force = distance * distance / k
                    attractive_forces[i] -= direction * force
                    attractive_forces[j] += direction * force

        return attractive_forces

    def _spring_layout(
        self, graph: chex.Array, num_nodes: int, seed: int = 42
    ) -> List[Tuple[float, float]]:
        """
        Compute a 2D spring layout for the given graph using
        the Fruchterman-Reingold force-directed algorithm.

        The algorithm computes a layout by simulating the graph as a physical system,
        where nodes are repelling each other and edges are attracting connected nodes.
        The method minimizes the energy of the system over several iterations.

        Args:
            graph: A Graph object representing the adjacency matrix of the graph.
            seed: An integer used to seed the random number generator for reproducibility.

        Returns:
            A list of tuples representing the 2D positions of nodes in the graph.
        """
        rng = np.random.default_rng(seed)
        pos = rng.random((num_nodes, 2)) * 2 - 1

        iterations = 100
        k = np.sqrt(5 / num_nodes)
        temperature = 2.0  # Added a temperature variable

        for _ in range(iterations):
            repulsive_forces = self._compute_repulsive_forces(
                np.zeros((num_nodes, 2)), pos, k, num_nodes
            )
            attractive_forces = self._compute_attractive_forces(
                graph, np.zeros((num_nodes, 2)), pos, k, num_nodes
            )

            pos += (repulsive_forces + attractive_forces) * temperature
            # Reduce the temperature (cooling factor) to refine the layout.
            temperature *= 0.9

            pos = np.clip(pos, -1, 1)  # Keep positions within the [-1, 1] range

        return [(float(p[0]), float(p[1])) for p in pos]

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, figsize=self.FIGURE_SIZE)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        if recreate:
            fig.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _render_nodes(
        self, ax: plt.Axes, pos: List[Tuple[float, float]], colors: chex.Array
    ) -> Tuple[List[plt.Circle], List[plt.Text]]:
        # Set the radius of the nodes as a fraction of the scale,
        # so nodes appear smaller when there are more of them.
        node_radius = 0.05 * 5 / self.node_scale
        circles = []
        labels = []

        for i, (x, y) in enumerate(pos):
            c = plt.Circle(
                (x, y),
                node_radius,
                color=self._color_mapping[colors[i]],
                fill=True,
                zorder=100,
            )
            circles.append(c)
            ax.add_artist(c)
            label = plt.Text(
                x,
                y,
                str(i),
                color="white",
                ha="center",
                va="center",
                weight="bold",
                zorder=200,
            )
            labels.append(label)
            ax.add_artist(label)

        return circles, labels

    def _render_edges(
        self,
        ax: plt.Axes,
        pos: List[Tuple[float, float]],
        adj_matrix: chex.Array,
        num_nodes: int,
    ) -> List[plt.Line2D]:
        edges = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge = plt.Line2D(
                    [pos[i][0], pos[j][0]],
                    [pos[i][1], pos[j][1]],
                    color=self._color_mapping[-1],
                    linewidth=0.5,
                    visible=adj_matrix[i, j],
                )
                ax.add_artist(edge)
                edges.append(edge)

        return edges

    def _calculate_node_scale(self, num_nodes: int) -> int:
        # Set the scale of the graph based on the number of nodes,
        # so the graph grows (at a decelerating rate) with more nodes.
        return 5 + int(np.sqrt(num_nodes))

    def _create_color_mapping(
        self,
        num_nodes: int,
    ) -> List[Tuple[float, float, float, float]]:
        colormap_indices = np.arange(0, 1, 1 / num_nodes)
        colormap = cm.get_cmap("hsv", num_nodes + 1)
        color_mapping = []
        for colormap_idx in colormap_indices:
            color_mapping.append(colormap(float(colormap_idx)))
        color_mapping.append((0.0, 0.0, 0.0, 1.0))  # Adding black to the color mapping
        return color_mapping
