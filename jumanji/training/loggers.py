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
import inspect
import logging
import pickle
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, DefaultDict, Dict, Optional, Type

import chex
import numpy as np
import omegaconf
import tensorboardX
from neptune import new as neptune


class Logger(AbstractContextManager):
    def __init__(
        self, save_checkpoint: bool, checkpoint_file_name: str = "training_state"
    ):
        self.save_checkpoint = save_checkpoint
        self.checkpoint_file_name = checkpoint_file_name

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

    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    def upload_checkpoint(self) -> None:
        """Uploads a checkpoint when exiting the logger."""

    def __enter__(self) -> "Logger":
        logging.info("Starting logger.")
        self._variables_enter = self._get_variables()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.save_checkpoint:
            self._variables_exit = self._get_variables()
            self._save_and_upload_checkpoint()
        logging.info("Closing logger...")
        self.close()

    def _save_and_upload_checkpoint(self) -> None:
        """Grabs the `training_state` variable from within the context manager, pickles it and
        saves it. This will break if the desired variable to checkpoint is not called
        `training_state`.
        """
        logging.info("Saving checkpoint...")
        in_context_variables = dict(
            set(self._variables_exit).difference(self._variables_enter)
        )
        variable_id = in_context_variables.get("training_state", None)
        if variable_id is not None:
            training_state = self._variables_exit[("training_state", variable_id)]
        else:
            training_state = None
            logging.debug(
                "Logger did not find variable 'training_state' at the context manager level."
            )
        with open(self.checkpoint_file_name, "wb") as file_:
            pickle.dump(training_state, file_)
        self.upload_checkpoint()
        logging.info(f"Checkpoint saved at '{self.checkpoint_file_name}'.")

    def _get_variables(self) -> Dict:
        """Returns the local variables that are accessible in the context of the context manager.
        This function gets the locals 2 stacks above. Index 0 is this very function, 1 is the
        __init__/__exit__ level, 2 is the context manager level.
        """
        return {(k, id(v)): v for k, v in inspect.stack()[2].frame.f_locals.items()}


class NoOpLogger(Logger):
    """Does nothing. This logger is useful in the case of multi-node training where only the
    master node should log.
    """

    def __init__(self) -> None:
        super().__init__(save_checkpoint=False)

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        pass


class TerminalLogger(Logger):
    """Logs to terminal."""

    def __init__(
        self, name: Optional[str] = None, save_checkpoint: bool = False
    ) -> None:
        super().__init__(save_checkpoint=save_checkpoint)
        if name:
            logging.info(f"Experiment: {name}.")

    def _format_values(self, data: Dict[str, Any]) -> str:
        return " | ".join(
            f"{key.replace('_', ' ').title()}: "
            f"{(f'{value:.3f}' if isinstance(value, (float, chex.Array)) else f'{value:,}')}"
            for key, value in sorted(data.items())
        )

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        if env_steps is not None:
            env_steps_str = f"Env Steps: {env_steps:.2e} | "
        else:
            env_steps_str = ""
        label_str = f"{label.replace('_', ' ').title()} >> " if label else ""
        logging.info(label_str + env_steps_str + self._format_values(data))


class ListLogger(Logger):
    """Logs to a dictionary of histories as lists."""

    def __init__(self, save_checkpoint: bool = False) -> None:
        super().__init__(save_checkpoint=save_checkpoint)
        self.history: DefaultDict = collections.defaultdict(list)

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        for key, value in data.items():
            self.history[key].append(value)


class TensorboardLogger(Logger):
    """Logs to tensorboard. To view logs, run a command like:
    tensorboard --logdir jumanji/training/outputs/{date}/{time}/{name}/
    """

    def __init__(self, name: str, save_checkpoint: bool = False) -> None:
        super().__init__(save_checkpoint=save_checkpoint)
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
        save_checkpoint: bool = False,
    ):
        super().__init__(save_checkpoint=save_checkpoint)
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

    def upload_checkpoint(self) -> None:
        self.run[f"checkpoint/{self.checkpoint_file_name}"].upload(
            self.checkpoint_file_name
        )
