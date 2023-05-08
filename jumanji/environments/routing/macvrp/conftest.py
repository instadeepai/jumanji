import pytest
from jumanji.environments.routing.macvrp.env import MACVRP

@pytest.fixture
def macvrp_env() -> MACVRP:
    """Instantiates a default MACVRP environment."""
    return MACVRP()