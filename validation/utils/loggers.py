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
from typing import Any, Dict

import numpy as np


class Logger(abc.ABC):
    """A logger has a `write` method."""

    def __init__(
        self,
        label: str = "",
    ):
        """Initializes the logger.

        Args:
          label: label string to use when logging.
        """
        self._label = label and f"[{label.replace('_', ' ').title()}] "

    @abc.abstractmethod
    def write(self, data: Dict[str, Any]) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str) -> None:
        self._label = label and f"[{label.replace('_', ' ').title()}] "


class NoOpLogger(Logger):
    """Simple Logger which does nothing and outputs no logs.

    This should be used sparingly, but it can prove useful if we want to quiet an
    individual component and have it produce no logging whatsoever.
    """

    def write(self, data: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class TerminalLogger(Logger):
    """Logs to terminal."""

    @staticmethod
    def _format_values(values: Dict[str, Any]) -> str:
        return " | ".join(
            f"{key.replace('_', ' ').title()} = "
            f"{(f'{value:.3f}' if isinstance(value, (float, np.number)) else str(value))}"
            for key, value in sorted(values.items())
        )

    def write(self, values: Dict[str, Any]) -> None:
        """Write a dictionary of metrics to the logger.

        Args:
            values: dictionary of metrics names and their values.

        """
        print(f"{self._label}{self._format_values(values)}")

    def close(self) -> None:
        pass
