import pytest
from jax import numpy as jnp

from jumanji.connect4.constants import BOARD_HEIGHT, BOARD_WIDTH


@pytest.fixture
def empty_board() -> jnp.array:
    """Empty board fixture (zeros everywhere).

    _____________________________
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    -----------------------------

    """
    return jnp.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.int8)
