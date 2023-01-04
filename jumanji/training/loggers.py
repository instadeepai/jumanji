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

import abc
import collections
import logging
from types import TracebackType
from typing import Any, DefaultDict, Dict, Optional, Type

import numpy as np


class Logger(abc.ABC):
    @abc.abstractmethod
    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        """Write a dictionary of metrics to the logger.

        Args:
            data: dictionary of metrics names and their values.
            label: optional label (e.g. 'train' or 'eval').
            env_steps: optional env step count.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    def __enter__(self) -> "Logger":
        logging.info("Starting logger.")
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        logging.info("Closing logger.")
        self.close()


class NoOpLogger(Logger):
    """Simple Logger which does nothing and outputs no logs.

    This should be used sparingly, but it can prove useful if we want to quiet an
    individual component and have it produce no logging whatsoever.
    """

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        pass

    def close(self) -> None:
        pass


class TerminalLogger(Logger):
    """Logs to terminal."""

    @staticmethod
    def _format_values(data: Dict[str, Any]) -> str:
        return " | ".join(
            f"{key.replace('_', ' ').title()} = "
            f"{(f'{value:.3f}' if isinstance(value, (float, np.number)) else str(value))}"
            for key, value in sorted(data.items())
        )

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        print(f"Env steps: {env_steps:.2e}", label, self._format_values(data))

    def close(self) -> None:
        pass


class ListLogger(Logger):
    """Logs to a dictionary of histories as lists."""

    def __init__(self) -> None:
        self.history: DefaultDict = collections.defaultdict(list)

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        for key, value in data.items():
            self.history[key].append(value)

    def close(self) -> None:
        pass
