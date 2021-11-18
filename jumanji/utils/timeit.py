import logging
import time
from contextlib import AbstractContextManager
from typing import Any


class TimeIt(AbstractContextManager):
    """
    Timer to use as a context manager to benchmark some computation.
    """

    def __init__(
        self, tag: str, frames: int = 0, ms: bool = False, print_: bool = True
    ):
        """

        Args:
            tag: title used for display.
            frames: number of computations during benchmarking.
            ms: True to have time displayed in milliseconds, False to have it in seconds.
            print_: True to print, False to not print but log with info level.
        """
        self.tag = tag
        self.frames = frames
        self.ms = ms
        self.print_ = print_
        self.msg = ""
        self.fps = 0.0

    def __enter__(self) -> "TimeIt":
        """Start the timer."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the timer and display time."""
        self.elapsed_secs = time.perf_counter() - self.start
        self.msg = f"{self.tag}: Elapsed time=" + (
            f"{1000*self.elapsed_secs:.2f}ms"
            if self.ms
            else f"{self.elapsed_secs:.2f}s"
        )
        if self.frames:
            self.fps = self.frames / self.elapsed_secs
            self.msg += f", Steps={self.frames}, FPS={self.fps:.2e}"
        if self.print_:
            print(self.msg)
        else:
            logging.info(self.msg)
