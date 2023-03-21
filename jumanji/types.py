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

from typing import TYPE_CHECKING, Dict, Generic, Optional, Sequence, TypeVar, Union

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import jax.numpy as jnp
from chex import Array


class StepType(jnp.int8):
    """Defines the status of a `TimeStep` within a sequence.

    First: 0
    Mid: 1
    Last: 2
    """

    # Denotes the first `TimeStep` in a sequence.
    FIRST = jnp.array(0, jnp.int8)
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = jnp.array(1, jnp.int8)
    # Denotes the last `TimeStep` in a sequence.
    LAST = jnp.array(2, jnp.int8)


Observation = TypeVar("Observation")


@dataclass
class TimeStep(Generic[Observation]):
    """Copied from `dm_env.TimeStep` with the goal of making it a Jax Type.
    The original `dm_env.TimeStep` is not a Jax type because inheriting a namedtuple is
    not treated as a valid Jax type (https://github.com/google/jax/issues/806).

    A `TimeStep` contains the data emitted by an environment at each step of
    interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
    NumPy array or a dict or list of arrays), and an associated `reward` and
    `discount`.

    The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
    `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
    have `StepType.MID.

    Attributes:
        step_type: A `StepType` enum value.
        reward:  A scalar, NumPy array, nested dict, list or tuple of rewards; or
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        discount: A scalar, NumPy array, nested dict, list or tuple of discount
            values in the range `[0, 1]`, or `None` if `step_type` is
            `StepType.FIRST`, i.e. at the start of a sequence.
        observation: A NumPy array, or a nested dict, list or tuple of arrays.
            Scalar values that can be cast to NumPy arrays (e.g. Python floats) are
            also valid in place of a scalar array.
        extras: environment metric(s) or information returned by the environment but
            not observed by the agent (hence not in the observation). For example, it
            could be whether an invalid action was taken. In most environments, extras
            is None.
    """

    step_type: StepType
    reward: Array
    discount: Array
    observation: Observation
    extras: Optional[Dict] = None

    def first(self) -> Array:
        return self.step_type == StepType.FIRST

    def mid(self) -> Array:
        return self.step_type == StepType.MID

    def last(self) -> Array:
        return self.step_type == StepType.LAST


def restart(
    observation: Observation,
    extras: Optional[Dict] = None,
    shape: Union[int, Sequence[int]] = (),
) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`.

    Args:
        observation: array or tree of arrays.
        extras: environment metric(s) or information returned by the environment but
            not observed by the agent (hence not in the observation). For example, it
            could be whether an invalid action was taken. In most environments, extras
            is None.
        shape: optional parameter to specify the shape of the rewards and discounts.
            Allows multi-agent environment compatibility. Defaults to () for
            scalar reward and discount.

    Returns:
        TimeStep identified as a reset.
    """
    return TimeStep(
        step_type=StepType.FIRST,
        reward=jnp.zeros(shape, dtype=float),
        discount=jnp.ones(shape, dtype=float),
        observation=observation,
        extras=extras,
    )


def transition(
    reward: Array,
    observation: Observation,
    discount: Optional[Array] = None,
    extras: Optional[Dict] = None,
    shape: Union[int, Sequence[int]] = (),
) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.MID`.

    Args:
        reward: array.
        observation: array or tree of arrays.
        discount: array.
        extras: environment metric(s) or information returned by the environment but
            not observed by the agent (hence not in the observation). For example, it
            could be whether an invalid action was taken. In most environments, extras
            is None.
        shape: optional parameter to specify the shape of the rewards and discounts.
            Allows multi-agent environment compatibility. Defaults to () for
            scalar reward and discount.

    Returns:
        TimeStep identified as a transition.
    """
    discount = discount if discount is not None else jnp.ones(shape, dtype=float)
    return TimeStep(
        step_type=StepType.MID,
        reward=reward,
        discount=discount,
        observation=observation,
        extras=extras,
    )


def termination(
    reward: Array,
    observation: Observation,
    extras: Optional[Dict] = None,
    shape: Union[int, Sequence[int]] = (),
) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

    Args:
        reward: array.
        observation: array or tree of arrays.
        extras: environment metric(s) or information returned by the environment but
            not observed by the agent (hence not in the observation). For example, it
            could be whether an invalid action was taken. In most environments, extras
            is None.
        shape: optional parameter to specify the shape of the rewards and discounts.
            Allows multi-agent environment compatibility. Defaults to () for
            scalar reward and discount.

    Returns:
        TimeStep identified as the termination of an episode.
    """
    return TimeStep(
        step_type=StepType.LAST,
        reward=reward,
        discount=jnp.zeros(shape, dtype=float),
        observation=observation,
        extras=extras,
    )


def truncation(
    reward: Array,
    observation: Observation,
    discount: Optional[Array] = None,
    extras: Optional[Dict] = None,
    shape: Union[int, Sequence[int]] = (),
) -> TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.LAST`.

    Args:
        reward: array.
        observation: array or tree of arrays.
        discount: array.
        extras: environment metric(s) or information returned by the environment but
            not observed by the agent (hence not in the observation). For example, it
            could be whether an invalid action was taken. In most environments, extras
            is None.
        shape: optional parameter to specify the shape of the rewards and discounts.
            Allows multi-agent environment compatibility. Defaults to () for
            scalar reward and discount.
    Returns:
        TimeStep identified as the truncation of an episode.
    """
    discount = discount if discount is not None else jnp.ones(shape, dtype=float)
    return TimeStep(
        step_type=StepType.LAST,
        reward=reward,
        discount=discount,
        observation=observation,
        extras=extras,
    )


def get_valid_dtype(dtype: Union[jnp.dtype, type]) -> jnp.dtype:
    """Cast a dtype taking into account the user type precision. E.g., if 64 bit is not enabled,
    jnp.dtype(jnp.float_) is still float64. By passing the given dtype through `jnp.empty` we get
    the supported dtype of float32.

    Args:
        dtype: jax numpy dtype or string specifying the array dtype.

    Returns:
        dtype converted to the correct type precision.
    """
    return jnp.empty((), dtype).dtype
