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

# Inspired from https://stackoverflow.com/questions/51849395/how-can-we-associate-a-python-context-m
# anager-to-the-variables-appearing-in-it#:~:text=also%20inspect%20the-,stack,-for%20locals()%20variables

import inspect
import logging
import time
from contextlib import AbstractContextManager
from typing import Any, Dict, Literal, Optional


class Timer(AbstractContextManager):
    def __init__(
        self,
        out_var_name: Optional[str] = None,
        num_steps_per_timing: Optional[int] = None,
    ):
        """Wraps some computation as a context manager. Expects the variable `out_var_name` to be
        newly created within the context of Timer and will append some timing metrics to it.

        Args:
            out_var_name: name of the variable to append timing metrics to.
            num_steps_per_timing: number of steps computed during the timing.
        """
        self.out_var_name = out_var_name
        self.num_steps_per_timing = num_steps_per_timing

    def _get_variables(self) -> Dict:
        """Returns the local variables that are accessible in the context of the context manager.
        This function gets the locals 2 stacks above. Index 0 is this very function, 1 is the
        __init__/__exit__ level, 2 is the context manager level.
        """
        return {(k, id(v)): v for k, v in inspect.stack()[2].frame.f_locals.items()}

    def __enter__(self) -> "Timer":
        self._variables_enter = self._get_variables()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        elapsed_time = time.perf_counter() - self._start_time
        self._variables_exit = self._get_variables()
        self.data = {"time": elapsed_time}
        if self.num_steps_per_timing is not None:
            self.data.update(
                steps_per_second=int(self.num_steps_per_timing / elapsed_time)
            )
        self._write_in_variable(self.data)
        return False

    def _write_in_variable(self, data: Dict[str, float]) -> None:
        in_context_variables = dict(
            set(self._variables_exit).difference(self._variables_enter)
        )
        metrics_id = in_context_variables.get(self.out_var_name, None)
        if metrics_id is not None:
            self._variables_exit[("metrics", metrics_id)].update(data)
        else:
            logging.debug(
                f"Timer did not find variable {self.out_var_name} in the context manager level."
            )
