import math
import time

from jumanji.utils.timeit import TimeIt


def test_timeit__init() -> None:
    """Validates initialization of TimeIt."""
    t = TimeIt(tag="Test", print_=False)
    assert isinstance(t, TimeIt)


def test_jax_environment_loop__run() -> None:
    """Validates TimeIt time measurement and FPS computation."""
    t = TimeIt(tag="Test", print_=False, frames=10)
    with t:
        time.sleep(0.01)
    # Check measured 0.01 s
    assert math.isclose(t.elapsed_secs, 0.01, abs_tol=0.005)
    # Check calculated 1000 frames/s (10 frames in 0.01 s)
    assert math.isclose(t.fps, 1000, rel_tol=0.1)
