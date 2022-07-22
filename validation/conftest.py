"""
Some common pytest fixtures that are often used in other testing files.
"""
import pytest

from jumanji.testing.fakes import (
    make_fake_agent,
    make_fake_dm_env,
    make_fake_environment,
)

fake_environment = pytest.fixture(make_fake_environment)
fake_dm_env = pytest.fixture(make_fake_dm_env)
fake_agent = pytest.fixture(make_fake_agent)
