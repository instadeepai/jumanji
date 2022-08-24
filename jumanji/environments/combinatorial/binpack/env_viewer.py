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

from typing import List, Tuple, Union

import matplotlib.cm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
import mpl_toolkits.mplot3d.axes3d

from jumanji.environments.combinatorial.binpack.types import State, item_from_space


class BinPackViewer:
    FONT_STYLE = "monospace"

    def __init__(self, name: str) -> None:
        """
        Viewer for the BinPack environment.

        Args:
            name: The window name to be used when initialising the window.
        """
        self._name = name

    def render(self, state: State) -> None:
        """
        Render the given state of the BinPack environment.

        Args:
            state: the State to render.
        """
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_state(state)
        self._update_ax(state)
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
        # Block for 2 seconds.
        fig.canvas.start_event_loop(2.0)

    def close(self) -> None:
        plt.close(self._name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name)
        if recreate:
            fig.set_tight_layout({"pad": False, "w_pad": 0.0, "h_pad": 0.0})
            fig.show()
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _add_state(self, state: State) -> None:
        n_items = len(state.items_mask)
        cmap = plt.cm.get_cmap("hsv", n_items)
        for i in range(n_items):
            if state.items_placed[i]:
                self._add_box(
                    (
                        state.items_location.x[i],
                        state.items_location.y[i],
                        state.items_location.z[i],
                    ),
                    (state.items.x_len[i], state.items.y_len[i], state.items.z_len[i]),
                    cmap(i),
                    0.3,
                )

        container = item_from_space(state.container)
        self._add_box(
            (0.0, 0.0, 0.0),
            (container.x_len, container.y_len, container.z_len),
            "cyan",
            0.05,
        )

    def _add_box(
        self,
        pos: Tuple[float, float, float],
        lens: Tuple[float, float, float],
        colour: Union[matplotlib.cm.ScalarMappable, str],
        alpha: float,
    ) -> None:
        """
        Add a box to the artist.

        Args:
            pos: (x, y, z)
            lens: lengths for the x, y and z dimensions.
            colour: colour of the box
            alpha: transparency of the box, 0 is completely transparent and 1 is opaque.

        """
        verts = self._create_box_vertices(pos, lens)
        poly3d = mpl_toolkits.mplot3d.art3d.Poly3DCollection(
            verts, linewidths=1, edgecolors="black", facecolors=colour, alpha=alpha
        )
        _, ax = self._get_fig_ax()
        ax.add_collection3d(poly3d)

    def _update_ax(self, state: State) -> None:
        """
        Sets the bounds of the scene and displays text about the scene.
        Args:
            state: State of the environment

        """
        fig, ax = self._get_fig_ax()
        eps = 0.05
        container = item_from_space(state.container)
        ax.set(
            xlim=(-container.x_len * eps, container.x_len * (1 + eps)),
            ylim=(-container.y_len * eps, container.y_len * (1 + eps)),
            zlim=(-container.z_len * eps, container.z_len * (1 + eps)),
        )
        ax.set_xlabel("x", font=self.FONT_STYLE)
        ax.set_ylabel("y", font=self.FONT_STYLE)
        ax.set_zlabel("z", font=self.FONT_STYLE)

        n_items = sum(state.items_mask)
        placed_items = sum(state.items_placed)
        container_volume = (
            float(container.x_len) * float(container.y_len) * float(container.z_len)
        )
        used_volume = self._get_used_volume(state)
        metrics = [
            ("Placed", f"{placed_items:{len(str(n_items))}}/{n_items}"),
            ("Used Volume", f"{used_volume / container_volume:6.1%}"),
        ]
        title = " | ".join(key + ": " + value for key, value in metrics)
        fig.suptitle(title, font=self.FONT_STYLE)

    def _create_box_vertices(
        self, pos: Tuple[float, float, float], lens: Tuple[float, float, float]
    ) -> List[List[Tuple[float, float, float]]]:
        """
        Args:
            pos: (x, y, z) of the corner closest to the origin of the coordinate space.
            lens: the x, y and z lengths of the box.

        Returns:
            For each face composing the box, a list containing the vertices that make that face.
        """
        verts = [
            (pos[0], pos[1], pos[2]),
            (pos[0] + lens[0], pos[1], pos[2]),
            (pos[0] + lens[0], pos[1] + lens[1], pos[2]),
            (pos[0] + lens[0], pos[1] + lens[1], pos[2] + lens[2]),
            (pos[0], pos[1] + lens[1], pos[2] + lens[2]),
            (pos[0], pos[1], pos[2] + lens[2]),
            (pos[0] + lens[0], pos[1], pos[2] + lens[2]),
            (pos[0], pos[1] + lens[1], pos[2]),
        ]
        faces = [
            [0, 1, 2, 7],
            [1, 2, 3, 6],
            [0, 1, 6, 5],
            [0, 7, 4, 5],
            [2, 7, 4, 3],
            [6, 3, 4, 5],
        ]
        return [[verts[i] for i in face] for face in faces]

    def _get_used_volume(self, state: State) -> float:
        used_volume = sum(
            float(state.items.x_len[i])
            * float(state.items.y_len[i])
            * float(state.items.z_len[i])
            for i, placed in enumerate(state.items_placed)
            if placed
        )
        return used_volume
