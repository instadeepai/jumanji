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
