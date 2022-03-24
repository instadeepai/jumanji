import pytest
from pyvirtualdisplay import Display

from jumanji.pcb_grid.pcb_grid import PcbGridEnv
from jumanji.pcb_grid.pcb_grid_viewer import PcbGridViewer


class TestViewer:
    width = height = 500

    @pytest.fixture(scope="module")
    def env(self) -> PcbGridEnv:
        """Creates the PCB grid environment."""
        env = PcbGridEnv(8, 8, 1)
        env.reset()

        return env

    @pytest.fixture(scope="module")
    def viewer(self, env: PcbGridEnv) -> PcbGridViewer:
        """Creates a viewer for the PCB grid environment."""
        return PcbGridViewer(
            env.num_agents, env.rows, env.cols, self.width, self.height
        )

    @pytest.fixture(scope="module")
    def display(self) -> Display:
        """Creates a virtual display so that a GUI is not displayed during testing."""
        display = Display(visible=False, size=(self.width, self.height))
        yield display.start()
        display.stop()

    def test_render(
        self, display: Display, env: PcbGridEnv, viewer: PcbGridViewer
    ) -> None:
        """Tests that the PcbGridViewer.render() method only raises a `ValueError`
        when given unsupported render modes.
        """
        viewer.render(env.grid, "human")
        viewer.render(env.grid, "fast")

        with pytest.raises(ValueError):
            viewer.render(env.grid, "abcdefg")

    def test_maybe_sleep(self, viewer: PcbGridViewer) -> None:
        """Tests that maybe_sleep throws a 'ValueError' when an unsupported mode is passed"""
        viewer.maybe_sleep("human")
        viewer.maybe_sleep("fast")

        with pytest.raises(ValueError):
            viewer.maybe_sleep("abcdefg")

    def test__draw_shape(
        self, display: Display, env: PcbGridEnv, viewer: PcbGridViewer
    ) -> None:
        """Tests that a `TypeError` is thrown when PcbGridViewer._draw_shape() is called with
        incorrect arguments.
        """
        viewer._draw_shape((1, 2, 3, 4), 1)

        with pytest.raises(TypeError):
            viewer._draw_shape((1, 2, 3), 1)  # type: ignore

        with pytest.raises(TypeError):
            viewer._draw_shape((1, 2, 3, 4), 1.5)  # type: ignore

        with pytest.raises(TypeError):
            viewer._draw_shape((1, 2, 3, 4, 6), 1)  # type: ignore

        viewer.close()
