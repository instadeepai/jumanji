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
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, DefaultDict, Dict, Optional, Type

import jax
import numpy as np
import omegaconf
import tensorboardX
from neptune import new as neptune


class Logger(AbstractContextManager):
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

    def __init__(self, name: Optional[str] = None) -> None:
        if name:
            logging.info(name)

    def _format_values(self, data: Dict[str, Any]) -> str:
        return " | ".join(
            f"{key.replace('_', ' ').title()}: "
            f"{(f'{value:.3f}' if isinstance(value, (float, jax.Array)) else f'{value:,}')}"
            for key, value in sorted(data.items())
        )

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        env_steps_str = f"Env steps: {env_steps:.2e} | " if env_steps else ""
        label_str = f"{label} >> " or ""
        logging.info(label_str + env_steps_str + self._format_values(data))

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


class TensorboardLogger(Logger):
    """Logs to tensorboard. To view logs, run a command like:
    tensorboard --logdir jumanji/training/outputs/{date}/{time}/{name}/
    """

    def __init__(self, name: str) -> None:
        if name:
            logging.info(name)
        self.writer = tensorboardX.SummaryWriter(logdir=name)
        self._env_steps = 0.0

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        if env_steps:
            self._env_steps = env_steps
        prefix = label and f"{label}/"
        for key, metric in data.items():
            if np.ndim(metric) == 0:
                if not np.isnan(metric):
                    self.writer.add_scalar(
                        tag=f"{prefix}/{key}",
                        scalar_value=metric,
                        global_step=int(self._env_steps),
                    )
            else:
                raise ValueError(f"Expected metric {key} to be a scalar, got {metric}.")

    def close(self) -> None:
        self.writer.close()


class NeptuneLogger(Logger):
    """Logs to the [neptune.ai](https://app.neptune.ai/) platform. The user is expected to have
    their NEPTUNE_API_TOKEN set as an environment variable. This can be done from the Neptune GUI.
    """

    def __init__(
        self,
        name: str,
        project: str,
        cfg: omegaconf.DictConfig,
    ):
        self.run = neptune.init_run(project=project, name=name)
        self.run["config"] = cfg
        self._env_steps = 0.0

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[float] = None,
    ) -> None:
        if env_steps:
            self._env_steps = env_steps
        prefix = label and f"{label}/"
        for key, metric in data.items():
            if np.ndim(metric) == 0:
                if not np.isnan(metric):
                    self.run[f"{prefix}/{key}"].log(
                        float(metric),
                        step=int(self._env_steps),
                        wait=True,
                    )
            else:
                raise ValueError(f"Expected metric {key} to be a scalar, got {metric}.")

    def close(self) -> None:
        self.run.stop()
