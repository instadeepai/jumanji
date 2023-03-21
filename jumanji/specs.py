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
import copy
import functools
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    NamedTuple,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import chex
import dm_env.specs
import gym
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.testing.pytrees import is_equal_pytree
from jumanji.types import get_valid_dtype

T = TypeVar("T")


class Spec(abc.ABC, Generic[T]):
    """Adapted from `dm_env.spec.Array`. This is an augmentation of the `Array` spec to allow for nested
    specs. `self.name`, `self.generate_value` and `self.validate` methods are adapted from the
    `dm_env` object."""

    def __init__(
        self,
        constructor: Union[Type[T], Callable[..., T]],
        name: str = "",
        **specs: "Spec",
    ):
        """Initializes a new spec.

        Args:
            constructor: the class or initialization function that creates the object represented
                by the spec.
            name: string containing a semantic name for the corresponding (nested) spec.
                Defaults to `''`.
            **specs: potential children specs each of which is either a nested spec or a primitive
                spec (`Array`, `BoundedArray`, etc). Importantly, the keywords used must exactly
                match the attribute names of the constructor.
        """
        self._name = name
        self._specs = specs
        self._constructor = constructor

        for spec_name, spec_value in specs.items():
            setattr(self, spec_name, spec_value)

    def __repr__(self) -> str:
        if self._specs.items():
            s = ""
            for spec_name, spec_value in self._specs.items():
                s += f"\t{spec_name}={spec_value},\n"
            return f"{self.name}(\n" + s + ")"
        return self.name

    @property
    def name(self) -> str:
        """Returns the name of the nested spec."""
        return self._name

    def validate(self, value: T) -> T:
        """Checks if a (potentially nested) value (tree of observations, actions...) conforms to
        this spec.

        Args:
            value: a (potentially nested) structure of jax arrays.

        Returns:
            value.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        if isinstance(value, tuple) and hasattr(value, "_asdict"):
            val = value._asdict()
        elif hasattr(value, "__dict__"):
            val = value.__dict__
        else:
            raise TypeError("The value provided must be a named tuple or a dataclass.")
        constructor_kwargs = jax.tree_util.tree_map(
            lambda spec, obs: spec.validate(obs), dict(self._specs), val
        )
        return self._constructor(**constructor_kwargs)

    def generate_value(self) -> T:
        """Generate a value which conforms to this spec."""
        constructor_kwargs = jax.tree_util.tree_map(
            lambda spec: spec.generate_value(), self._specs
        )
        return self._constructor(**constructor_kwargs)

    def replace(self, **kwargs: Any) -> "Spec":
        """Returns a new copy of `self` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `self`.
        """
        dict_copy = copy.deepcopy(self._specs)
        dict_copy.update(kwargs)
        return Spec(self._constructor, self.name, **dict_copy)

    def __eq__(self, other: "Spec") -> bool:  # type: ignore[override]
        if not isinstance(other, Spec):
            return NotImplemented
        return is_equal_pytree(self._specs, other._specs)

    def __getitem__(self, item: str) -> "Spec":
        return self._specs[item]


class Array(Spec[chex.Array]):
    """Describes a jax array spec. This is adapted from `dm_env.specs.Array` to suit Jax environments.

    An `Array` spec allows an API to describe the arrays that it accepts or returns, before that
    array exists.
    """

    def __init__(self, shape: Iterable, dtype: Union[jnp.dtype, type], name: str = ""):
        """Initializes a new `Array` spec.

        Args:
            shape: an iterable specifying the array shape.
            dtype: jax numpy dtype or string specifying the array dtype.
            name: string containing a semantic name for the corresponding array. Defaults to `''`.
        """
        self._constructor = lambda: jnp.zeros(shape, dtype)
        super().__init__(constructor=self._constructor, name=name)
        self._shape = tuple(int(dim) for dim in shape)
        self._dtype = get_valid_dtype(dtype)

    def __repr__(self) -> str:
        return f"Array(shape={repr(self.shape)}, dtype={repr(self.dtype)}, name={repr(self.name)})"

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return Array, (self._shape, self._dtype, self._name)

    @property
    def shape(self) -> Tuple:
        """Returns a `tuple` specifying the array shape."""
        return self._shape

    @property
    def dtype(self) -> jnp.dtype:
        """Returns a jax numpy dtype specifying the array dtype."""
        return self._dtype

    def _fail_validation(self, message: str) -> None:
        if self.name:
            message += f" for spec {self.name}."
        else:
            message += "."
        raise ValueError(message)

    def validate(self, value: chex.Numeric) -> chex.Array:
        """Checks if value conforms to this spec.

        Args:
            value: a jax array or value convertible to one via `jnp.asarray`.

        Returns:
            value, converted if necessary to a jax array.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        value = jnp.asarray(value)
        if value.shape != self.shape:
            self._fail_validation(
                f"Expected shape {self.shape} but found {value.shape}"
            )
        if value.dtype != self.dtype:
            self._fail_validation(
                f"Expected dtype {self.dtype} but found {value.dtype}"
            )
        return value

    def _get_constructor_kwargs(self) -> Dict[str, Any]:
        """Returns constructor kwargs for instantiating a new copy of this spec."""
        # Get the names and kinds of the constructor parameters.
        params = inspect.signature(
            functools.partial(type(self).__init__, self)
        ).parameters
        # __init__ must not accept *args or **kwargs, since otherwise we won't be
        # able to infer what the corresponding attribute names are.
        kinds = {value.kind for value in params.values()}
        if inspect.Parameter.VAR_POSITIONAL in kinds:
            raise TypeError("specs.Array subclasses must not accept *args.")
        elif inspect.Parameter.VAR_KEYWORD in kinds:
            raise TypeError("specs.Array subclasses must not accept **kwargs.")
        # Note that we assume direct correspondence between the names of constructor
        # arguments and attributes.
        return {name: getattr(self, name) for name in params.keys()}

    def replace(self, **kwargs: Any) -> "Array":
        """Returns a new copy of `self` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `self`.
        """
        all_kwargs = self._get_constructor_kwargs()
        all_kwargs.update(kwargs)
        return type(self)(**all_kwargs)

    def __eq__(self, other: "Array") -> bool:  # type: ignore[override]
        if not isinstance(other, Array):
            return NotImplemented
        return (
            (self.shape == other.shape)
            and (self.dtype == other.dtype)
            and (self.name == other.name)
        )


class BoundedArray(Array):
    """Bounded array spec that specifies minimum and maximum values for an environment. This is
    adapted from `dm_env.specs.BoundedArray` to suit Jax environments.

    Example usage:
    ```python
    # Specifying the same minimum and maximum for every element.
    spec = BoundedArray((3, 4), float, minimum=0.0, maximum=1.0)

    # Specifying a different minimum and maximum for each element.
    spec = BoundedArray((2,), float, minimum=[0.1, 0.2], maximum=[0.9, 0.9])

    # Specifying the same minimum and a different maximum for each element.
    spec = BoundedArray((3,), float, minimum=-10.0, maximum=[4.0, 5.0, 3.0])
    ```

    Bounds are meant to be inclusive. This is especially important for integer types. The following
    spec will be satisfied by arrays with values in the set {0, 1, 2}:
    ```python
    spec = BoundedArray((3, 4), int, minimum=0, maximum=2)
    ```

    Note that one or both bounds may be infinite. For example, the set of non-negative floats can be
    expressed as:
    ```python
    spec = BoundedArray((), float, minimum=0.0, maximum=jnp.inf)
    ```
    In this case `jnp.inf` would be considered valid, since the upper bound is inclusive.
    """

    def __init__(
        self,
        shape: Iterable,
        dtype: Union[jnp.dtype, type],
        minimum: Union[chex.Numeric, Sequence],
        maximum: Union[chex.Numeric, Sequence],
        name: str = "",
    ):
        """
        Args:
            shape: an iterable specifying the array shape.
            dtype: jax numpy dtype or string specifying the array dtype.
            minimum: number or sequence specifying the minimum element bounds (inclusive).
                Must be broadcastable to `shape`.
            maximum: number or sequence specifying the maximum element bounds (inclusive).
                Must be broadcastable to `shape`.
            name: string containing a semantic name for the corresponding array. Defaults to `''`.

        Raises:
            ValueError: if `minimum` or `maximum` are not broadcastable to `shape`.
            ValueError: if any values in `minimum` are greater than their corresponding value
                in `maximum`.
            TypeError: if the shape is not an iterable or if the `dtype` is an invalid jax numpy
                dtype.
        """
        super().__init__(shape, dtype, name)
        minimum = jnp.asarray(minimum, dtype)
        maximum = jnp.asarray(maximum, dtype)
        try:
            bcast_minimum = jnp.broadcast_to(minimum, shape=shape)
        except ValueError as jnp_exception:
            raise ValueError(
                "`minimum` is incompatible with `shape`"
            ) from jnp_exception
        try:
            bcast_maximum = jnp.broadcast_to(maximum, shape=shape)
        except ValueError as jnp_exception:
            raise ValueError(
                "`maximum` is incompatible with `shape`"
            ) from jnp_exception

        if jnp.any(bcast_minimum > bcast_maximum):
            raise ValueError(
                f"All values in `minimum` must be less than or equal to their corresponding "
                f"value in `maximum`, got: \n\tminimum={repr(minimum)}\n\tmaximum={repr(maximum)}"
            )
        self._constructor = lambda: jnp.full(shape, minimum, dtype)
        self._minimum = minimum
        self._maximum = maximum

    def __repr__(self) -> str:
        return (
            f"BoundedArray(shape={repr(self.shape)}, dtype={repr(self.dtype)}, "
            f"name={repr(self.name)}, minimum={repr(self.minimum)}, maximum={repr(self.maximum)})"
        )

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return BoundedArray, (
            self._shape,
            self._dtype,
            self._minimum,
            self._maximum,
            self._name,
        )

    @property
    def minimum(self) -> chex.Array:
        """Returns a Jax array specifying the minimum bounds (inclusive)."""
        return self._minimum

    @property
    def maximum(self) -> chex.Array:
        """Returns a Jax array specifying the maximum bounds (inclusive)."""
        return self._maximum

    def validate(self, value: chex.Numeric) -> chex.Array:
        value = super().validate(value)
        if (value < self.minimum).any() or (value > self.maximum).any():
            self._fail_validation(
                "Values were not all within bounds "
                f"{repr(self.minimum)} <= {repr(value)} <= {repr(self.maximum)}"
            )
        return value

    def __eq__(self, other: "BoundedArray") -> bool:  # type: ignore[override]
        if not isinstance(other, BoundedArray):
            return NotImplemented
        return (
            (self.shape == other.shape)
            and (self.dtype == other.dtype)
            and (self.minimum == other.minimum)
            and (self.maximum == other.maximum)
            and (self.name == other.name)
        )


class DiscreteArray(BoundedArray):
    """Represents a discrete, scalar, zero-based space. This is adapted from
    `dm_env.specs`.BoundedArray to suit Jax environments.

    This is a special case of the parent `BoundedArray` class. It represents a 0-dimensional jax
    array  containing a single integer value between 0 and num_values - 1 (inclusive), and exposes
    a scalar `num_values` property in addition to the standard `BoundedArray` interface.

    For an example use-case, this can be used to define the action space of a simple RL environment
    that accepts discrete actions.
    """

    def __init__(
        self, num_values: int, dtype: Union[jnp.dtype, type] = jnp.int32, name: str = ""
    ):
        """Initializes a new `DiscreteArray` spec.

        Args:
            num_values: integer specifying the number of possible values to represent.
            dtype: the dtype of the jax array. Must be an integer type.
            name: string containing a semantic name for the corresponding array. Defaults to `''`.

        Raises:
            ValueError: if `num_values` is not positive, if `dtype` is not integer.
        """
        if num_values <= 0 or not jnp.issubdtype(type(num_values), jnp.integer):
            raise ValueError(
                f"`num_values` must be a positive integer, got {num_values}."
            )

        if not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"`dtype` must be integer, got {dtype}.")

        num_values = int(num_values)
        maximum = num_values - 1
        super().__init__(shape=(), dtype=dtype, minimum=0, maximum=maximum, name=name)
        self._num_values = num_values

    def __repr__(self) -> str:
        return (
            f"DiscreteArray(shape={repr(self.shape)}, dtype={repr(self.dtype)}, "
            f"name={repr(self.name)}, minimum={repr(self.minimum)}, maximum={repr(self.maximum)}, "
            f"num_values={repr(self.num_values)})"
        )

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return DiscreteArray, (self._num_values, self._dtype, self._name)

    @property
    def num_values(self) -> int:
        """Returns the number of items."""
        return self._num_values

    def __eq__(self, other: "DiscreteArray") -> bool:  # type: ignore[override]
        if not isinstance(other, DiscreteArray):
            return NotImplemented
        return (
            (self.num_values == other.num_values)
            and (self.dtype == other.dtype)
            and (self.name == other.name)
        )


class MultiDiscreteArray(BoundedArray):
    """Generalizes DiscreteArray to a multi-dimensional array (e.g. a vector of actions) similarly
    to Gym's MultiDiscrete Space. This is commonly used for the action spec in Jumanji.
    """

    def __init__(
        self,
        num_values: chex.Array,
        dtype: Union[jnp.dtype, type] = jnp.int32,
        name: str = "",
    ):
        """Initializes a new `MultiDiscreteArray` spec.

        Args:
            num_values: Array of integers specifying the number of possible values to represent for
                each element of the action space.
            dtype: the dtype of the jax array. Must be an integer type.
            name: string containing a semantic name for the corresponding array. Defaults to `''`.

        Raises:
            ValueError: if `num_values` are not all positive, if `dtype` is not integer.
        """
        if (num_values <= 0).any() or not jnp.issubdtype(num_values.dtype, jnp.integer):
            raise ValueError(
                f"`num_values` must be an array of positive integers, got {num_values}."
            )

        if not jnp.issubdtype(dtype, jnp.integer):
            raise ValueError(f"`dtype` must be integer, got {dtype}.")

        num_values = num_values
        maximum = num_values - 1
        super().__init__(
            shape=num_values.shape,
            dtype=dtype,
            minimum=jnp.zeros_like(num_values),
            maximum=maximum,
            name=name,
        )
        self._num_values = num_values

    def __repr__(self) -> str:
        return (
            f"MultiDiscreteArray(shape={repr(self.shape)}, dtype={repr(self.dtype)}, "
            f"name={repr(self.name)}, minimum={repr(self.minimum)}, maximum={repr(self.maximum)}, "
            f"num_values={repr(self.num_values)})"
        )

    def __reduce__(self) -> Any:
        """To allow pickle to serialize the spec."""
        return MultiDiscreteArray, (self._num_values, self._dtype, self._name)

    @property
    def num_values(self) -> chex.Array:
        """Returns the number of possible values for each element of the action vector."""
        return self._num_values

    def __eq__(self, other: "MultiDiscreteArray") -> bool:  # type: ignore[override]
        if not isinstance(other, MultiDiscreteArray):
            return NotImplemented
        return (
            (self.num_values == other.num_values).all()
            and (self.dtype == other.dtype)
            and (self.name == other.name)
        )


def jumanji_specs_to_dm_env_specs(
    spec: Spec,
) -> Union[
    dm_env.specs.DiscreteArray,
    dm_env.specs.BoundedArray,
    dm_env.specs.Array,
    Dict[str, dm_env.specs.Array],
]:
    """Converts jumanji specs to dm_env specs or a tree of dm_env specs.

    Args:
        spec: jumanji spec of type `jumanji.specs.Spec`.

    Returns:
        `dm_env.specs.Array` or pytree of `dm_env.specs.Array` corresponding to the equivalent
            jumanji specs implementation.
    """
    if isinstance(spec, DiscreteArray):
        return dm_env.specs.DiscreteArray(
            num_values=spec.num_values,
            dtype=spec.dtype,
            name=spec.name if spec.name else None,
        )
    elif isinstance(spec, BoundedArray):
        return dm_env.specs.BoundedArray(
            shape=spec.shape,
            dtype=spec.dtype,
            minimum=spec.minimum,
            maximum=spec.maximum,
            name=spec.name if spec.name else None,
        )
    elif isinstance(spec, Array):
        return dm_env.specs.Array(
            shape=spec.shape,
            dtype=spec.dtype,
            name=spec.name if spec.name else None,
        )
    else:
        # Nested spec
        return {
            # Iterate over specs
            f"{key}": jumanji_specs_to_dm_env_specs(value)
            for key, value in vars(spec).items()
            if isinstance(value, Spec)
        }


def jumanji_specs_to_gym_spaces(
    spec: Spec,
) -> Union[
    gym.spaces.Box,
    gym.spaces.Discrete,
    gym.spaces.MultiDiscrete,
    gym.spaces.Space,
    gym.spaces.Dict,
]:
    """Converts jumanji specs to gym spaces.

    Args:
        spec: jumanji spec of type jumanji.specs.Spec, can be an Array or any nested spec.

    Returns:
        gym.spaces object corresponding to the equivalent jumanji specs implementation.
    """
    if isinstance(spec, DiscreteArray):
        return gym.spaces.Discrete(n=spec.num_values, seed=None)
    elif isinstance(spec, MultiDiscreteArray):
        return gym.spaces.MultiDiscrete(nvec=spec.num_values, seed=None)
    elif isinstance(spec, BoundedArray):
        # When using NumPy: 1.21.5:
        # MyPy error: "Call to untyped function "broadcast_to" in typed context"
        low = np.broadcast_to(spec.minimum, shape=spec.shape)  # type: ignore
        high = np.broadcast_to(spec.maximum, shape=spec.shape)  # type: ignore
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=spec.shape,
            dtype=spec.dtype,
        )
    elif isinstance(spec, Array):
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=spec.shape,
            dtype=spec.dtype,
            seed=None,
        )
    else:
        # Nested spec
        return gym.spaces.Dict(
            {
                # Iterate over specs
                f"{key}": jumanji_specs_to_gym_spaces(value)
                for key, value in vars(spec).items()
                if isinstance(value, Spec)
            }
        )


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""

    observations: Spec
    actions: Spec
    rewards: Array
    discounts: BoundedArray
