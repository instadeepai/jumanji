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

import time
from contextlib import AbstractContextManager
from typing import Any, Dict

from validation.utils import loggers


class TimeIt(AbstractContextManager):
    """
    Timer to use as a context manager to benchmark some computation.
    """

    def __init__(
        self,
        logger: loggers.Logger,
        frames: int = 0,
        ms: bool = False,
    ):
        """

        Args:
            logger: logger to print run time.
            frames: number of computations during benchmarking.
            ms: True to have time displayed in milliseconds, False to have it in seconds.
        """
        self.frames = frames
        self.ms = ms
        self.logger = logger
        self.msg = ""
        self.fps = 0.0

    def __enter__(self) -> "TimeIt":
        """Start the timer."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer and display time."""
        self.elapsed_secs = time.perf_counter() - self.start
        values: Dict = {}
        if self.ms:
            values.update(elapsed_time=f"{1000 * self.elapsed_secs:.2f} ms")
        else:
            values.update(elapsed_time=f"{self.elapsed_secs:.2f} s")
        if self.frames:
            self.fps = self.frames / self.elapsed_secs
            self.msg += f" | Steps = {self.frames} | FPS = {self.fps:.2e}"
            values.update(steps=self.frames, fps=self.fps)
        self.logger.write(values)
