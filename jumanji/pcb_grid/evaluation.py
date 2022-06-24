import numpy as np

from jumanji.pcb_grid import PcbGridEnv
from jumanji.pcb_grid.pcb_grid import SOURCE


def wire_length(env: PcbGridEnv) -> int:
    """Calculates the length of all the wires on the grid."""
    return sum(
        [np.count_nonzero(env.grid == SOURCE + 3 * i) for i in range(env.num_agents)]
    )


def proportion_connected(env: PcbGridEnv) -> float:
    """Calculates the proportion of wires that are connected."""
    return sum([agent.done for agent in env.agents]) / env.num_agents


def is_board_complete(env: PcbGridEnv) -> bool:
    """True if all wires connect, otherwise false."""
    return sum([agent.done for agent in env.agents]) == env.num_agents
