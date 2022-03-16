import pytest

from jumanji.pcb_grid import PcbGridEnv
from jumanji.pcb_grid.evaluation import (
    is_board_complete,
    proportion_connected,
    wire_length,
)
from jumanji.pcb_grid.pcb_grid import NOOP


class TestEvaluation:
    @pytest.fixture(scope="module")
    def env(self) -> PcbGridEnv:
        """Creates the PCB grid environment."""
        env = PcbGridEnv(8, 8, 2)
        env.reset()
        return env

    def test_evaluation__is_board_complete(self, env: PcbGridEnv) -> None:
        """Tests evaluation method is_board_complete correctly returns True when agents have
        reached desired positions."""
        env.reset()
        assert not is_board_complete(env)
        env.agents[0].position = env.agents[0].target
        assert not is_board_complete(env)
        env.agents[1].position = env.agents[1].target
        assert is_board_complete(env)

    def test_evaluation__proportion_connected(self, env: PcbGridEnv) -> None:
        """Tests that proportion_connected returns the correct value when different numbers of
        agents are connected."""
        env.reset()
        assert proportion_connected(env) == 0.0
        env.agents[0].position = env.agents[0].target
        assert proportion_connected(env) == 0.5
        env.agents[1].position = env.agents[1].target
        assert proportion_connected(env) == 1.0

    def test_evaluation__wire_length(self, env: PcbGridEnv) -> None:
        """Tests that `wire_length` accurately counts the number of wires on the board."""
        obs = env.reset()
        assert wire_length(env) == 0
        action = obs[0]["action_mask"][1:].index(1) + 1
        env.step({0: action, 1: NOOP})
        assert wire_length(env) == 1
