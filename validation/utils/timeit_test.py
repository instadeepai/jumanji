import math
import time

from pytest import CaptureFixture

from validation.utils import loggers
from validation.utils.timeit import TimeIt


def test_timeit__init() -> None:
    """Validates initialization of TimeIt."""
    t = TimeIt(logger=loggers.NoOpLogger())
    assert isinstance(t, TimeIt)


def test_environment_loop__run(capsys: CaptureFixture) -> None:
    """Validates TimeIt time measurement and FPS computation."""
    with TimeIt(logger=loggers.TerminalLogger("Test"), frames=10) as t:
        time.sleep(0.01)
    # Check measured 0.01 s
    assert math.isclose(t.elapsed_secs, 0.01, abs_tol=0.005)
    # Check calculated 1000 frames/s (10 frames in 0.01 s)
    assert math.isclose(t.fps, 1000, rel_tol=0.1)
    assert capsys.readouterr().out
