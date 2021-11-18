"""
Some common pytest fixtures that are often used in other testing files.
"""
import pytest

from jumanji.testing.fakes import make_fake_dm_env, make_fake_jax_env

fake_jax_env = pytest.fixture(make_fake_jax_env)
fake_dm_env = pytest.fixture(make_fake_dm_env)
