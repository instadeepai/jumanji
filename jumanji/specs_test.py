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

# Adapted from dm_env.specs_test
# ============================================================================
import pickle
from typing import Any, Iterable, Sequence, Tuple, Union

import chex
import dm_env.specs
import jax.numpy as jnp
import pytest

from jumanji import specs


class NestedSpecOneArray(specs.Spec[Tuple[chex.Array, ...]]):
    """An example of nested Spec that is a tuple of one Array."""

    def __init__(self, shape: Iterable, dtype: Union[jnp.dtype, type], name: str = ""):
        super(NestedSpecOneArray, self).__init__(name)
        self._shape = tuple(int(dim) for dim in shape)
        self._dtype = jnp.dtype(dtype)
        self.array_spec = specs.Array(self._shape, self._dtype)

    def __repr__(self) -> str:
        return self.name

    def validate(
        self,
        value: Tuple[chex.Array, ...],
    ) -> Tuple[chex.Array, ...]:
        array = value[0]
        array = self.array_spec.validate(array)
        return (array,)

    def generate_value(
        self,
    ) -> Tuple[chex.Array, ...]:
        return (self.array_spec.generate_value(),)

    def replace(self, **kwargs: Any) -> "NestedSpecOneArray":
        new_array_spec = self.array_spec.replace(**kwargs)
        return NestedSpecOneArray(
            new_array_spec.shape, new_array_spec.dtype, new_array_spec.name
        )


@pytest.fixture
def nested_spec_one_array(
    shape: Iterable = (2, 3),
    dtype: Union[jnp.dtype, type] = jnp.float32,
    name: str = "",
) -> NestedSpecOneArray:
    return NestedSpecOneArray(shape, dtype, name)


class TestArray:
    def test_shape_type_error(self) -> None:
        with pytest.raises(TypeError):
            specs.Array(32, jnp.int_)  # type: ignore

    def test_shape_element_type_error(self) -> None:
        with pytest.raises(TypeError):
            specs.Array([None], jnp.int_)

    def test_dtype_type_error(self) -> None:
        with pytest.raises(TypeError):
            specs.Array((1, 2, 3), "32")

    def test_scalar_shape(self) -> None:
        specs.Array((), jnp.int32)

    def test_string_dtype_error(self) -> None:
        specs.Array((1, 2, 3), "int32")

    def test_dtype(self) -> None:
        specs.Array((1, 2, 3), jnp.int_)

    def test_correct_dtype(self) -> None:
        spec = specs.Array((1, 2, 3), jnp.int32)
        assert spec.dtype == jnp.int32

    def test_shape(self) -> None:
        spec = specs.Array([1, 2, 3], jnp.int_)
        assert spec.shape == (1, 2, 3)

    @pytest.mark.parametrize(
        "value, is_valid",
        [
            (jnp.zeros((1, 2), dtype=jnp.int32), True),
            (jnp.zeros((1, 2), dtype=jnp.float32), False),
        ],
    )
    def test_validate_dtype(self, value: chex.Numeric, is_valid: bool) -> None:
        spec = specs.Array((1, 2), jnp.int32)
        if is_valid:  # Should not raise any exception.
            spec.validate(value)
        else:
            with pytest.raises(ValueError):
                spec.validate(value)

    @pytest.mark.parametrize(
        "value, is_valid",
        [
            (jnp.zeros((1, 2), dtype=jnp.int32), True),
            (jnp.zeros((1, 2, 3), dtype=jnp.int32), False),
        ],
    )
    def test_validate_shape(self, value: chex.Numeric, is_valid: bool) -> None:
        spec = specs.Array((1, 2), jnp.int32)
        if is_valid:  # Should not raise any exception.
            spec.validate(value)
        else:
            with pytest.raises(ValueError):
                spec.validate(value)

    def test_generate_value(self) -> None:
        spec = specs.Array((1, 2), jnp.int32)
        test_value = spec.generate_value()
        spec.validate(test_value)

    def test_serialization(self) -> None:
        spec = specs.Array([1, 5], jnp.float32, "pickle_test")
        loaded_spec = pickle.loads(pickle.dumps(spec))
        assert isinstance(loaded_spec, spec.__class__)
        assert loaded_spec.dtype == spec.dtype
        assert loaded_spec.shape == spec.shape
        assert loaded_spec.name == spec.name

    @pytest.mark.parametrize(
        "arg_name, new_value",
        [
            ("shape", (2, 3)),
            ("dtype", jnp.int32),
            ("name", "something_else"),
        ],
    )
    def test_replace(self, arg_name: str, new_value: Any) -> None:
        old_spec = specs.Array([1, 5], jnp.float32, "test")
        new_spec = old_spec.replace(**{arg_name: new_value})
        assert new_spec is not old_spec
        assert getattr(new_spec, arg_name) == new_value
        for attr_name in {"shape", "dtype", "name"}.difference([arg_name]):
            assert getattr(new_spec, attr_name) == getattr(old_spec, attr_name)


class TestBoundedArray:
    def test_invalid_minimum(self) -> None:
        with pytest.raises(ValueError):
            specs.BoundedArray((3, 5), jnp.uint8, (0, 0, 0), (1, 1))

    def test_invalid_maximum(self) -> None:
        with pytest.raises(ValueError):
            specs.BoundedArray((3, 5), jnp.uint8, 0, (1, 1, 1))

    def test_min_max_attributes(self) -> None:
        spec = specs.BoundedArray((1, 2, 3), jnp.float32, 0, (5, 5, 5))
        assert isinstance(spec.minimum, jnp.ndarray)
        assert isinstance(spec.maximum, jnp.ndarray)

    @pytest.mark.parametrize(
        "spec_dtype, min_dtype, max_dtype",
        [
            (jnp.float32, jnp.float64, jnp.int32),
            (jnp.uint32, jnp.uint8, float),
        ],
    )
    def test_min_max_casting(
        self, spec_dtype: jnp.dtype, min_dtype: jnp.dtype, max_dtype: jnp.dtype
    ) -> None:
        minimum = jnp.array(0.0, dtype=min_dtype)
        maximum = jnp.array((3.14, 15.9, 265.4), dtype=max_dtype)
        spec = specs.BoundedArray(
            shape=(1, 2, 3), dtype=spec_dtype, minimum=minimum, maximum=maximum
        )
        assert spec.minimum.dtype == spec_dtype
        assert spec.maximum.dtype == spec_dtype

    def test_read_only(self) -> None:
        spec = specs.BoundedArray((1, 2, 3), jnp.float32, 0, (5, 5, 5))
        with pytest.raises(TypeError):
            spec.minimum[0] = -1
        with pytest.raises(TypeError):
            spec.maximum[0] = 100

    def test_equal_broadcasting_bounds(self) -> None:
        spec_1 = specs.BoundedArray((1, 2), jnp.float32, minimum=0.0, maximum=1.0)
        spec_2 = specs.BoundedArray(
            (1, 2), jnp.float32, minimum=[0.0, 0.0], maximum=[1.0, 1.0]
        )
        assert jnp.all(spec_1.minimum == spec_2.minimum)
        assert jnp.all(spec_1.maximum == spec_2.maximum)

    def test_not_equal_different_minimum(self) -> None:
        spec_1 = specs.BoundedArray(
            (1, 2), jnp.float32, minimum=[0.0, -0.6], maximum=[1.0, 1.0]
        )
        spec_2 = specs.BoundedArray(
            (1, 2), jnp.float32, minimum=[0.0, 0.0], maximum=[1.0, 1.0]
        )
        assert not jnp.all(spec_1.minimum == spec_2.minimum)
        assert jnp.all(spec_1.maximum == spec_2.maximum)

    def test_not_equal_different_maximum(self) -> None:
        spec_1 = specs.BoundedArray((1, 2), jnp.int32, minimum=0.0, maximum=2.0)
        spec_2 = specs.BoundedArray(
            (1, 2), jnp.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0]
        )
        assert not jnp.all(spec_1.maximum == spec_2.maximum)
        assert jnp.all(spec_1.minimum == spec_2.minimum)

    def test_repr(self) -> None:
        as_string = repr(
            specs.BoundedArray((1, 2), jnp.int32, minimum=73.0, maximum=101.0)
        )
        assert "73" in as_string
        assert "101" in as_string

    @pytest.mark.parametrize(
        "value, is_valid",
        [
            (jnp.array([[5, 6], [8, 10]], dtype=jnp.int32), True),
            (jnp.array([[5, 6], [8, 11]], dtype=jnp.int32), False),
            (jnp.array([[4, 6], [8, 10]], dtype=jnp.int32), False),
        ],
    )
    def test_validate_bounds(self, value: chex.Numeric, is_valid: bool) -> None:
        spec = specs.BoundedArray((2, 2), jnp.int32, minimum=5, maximum=10)
        if is_valid:  # Should not raise any exception.
            spec.validate(value)
        else:
            with pytest.raises(ValueError):
                spec.validate(value)

    @pytest.mark.parametrize(
        "minimum, maximum, value, is_valid",
        [
            # Semi-infinite intervals.
            (0.0, jnp.inf, 0.0, True),
            (0.0, jnp.inf, 1.0, True),
            (0.0, jnp.inf, jnp.inf, True),
            (0.0, jnp.inf, -1.0, False),
            (0.0, jnp.inf, -jnp.inf, False),
            (-jnp.inf, 0.0, 0.0, True),
            (-jnp.inf, 0.0, -1.0, True),
            (-jnp.inf, 0.0, -jnp.inf, True),
            (-jnp.inf, 0.0, 1.0, False),
            # Infinite interval.
            (-jnp.inf, jnp.inf, 1.0, True),
            (-jnp.inf, jnp.inf, -1.0, True),
            (-jnp.inf, jnp.inf, -jnp.inf, True),
            (-jnp.inf, jnp.inf, jnp.inf, True),
            # Special case where minimum == maximum.
            (0.0, 0.0, 0.0, True),
            (0.0, 0.0, jnp.finfo(float).eps, False),
        ],
    )
    def test_validate_bounds_float(
        self,
        minimum: Union[chex.Numeric, Sequence],
        maximum: Union[chex.Numeric, Sequence],
        value: chex.Numeric,
        is_valid: bool,
    ) -> None:
        spec = specs.BoundedArray((), jnp.float32, minimum=minimum, maximum=maximum)
        if is_valid:  # Should not raise any exception.
            spec.validate(value)
        else:
            with pytest.raises(ValueError):
                spec.validate(value)

    def test_validate_returns_value(self) -> None:
        spec = specs.BoundedArray([1], jnp.int32, minimum=0, maximum=1)
        validated_value = spec.validate(jnp.array([0], dtype=jnp.int32))
        assert validated_value is not None

    def test_generate_value(self) -> None:
        spec = specs.BoundedArray((2, 2), jnp.int32, minimum=5, maximum=10)
        test_value = spec.generate_value()
        spec.validate(test_value)

    def test_scalar_bounds(self) -> None:
        spec = specs.BoundedArray((), jnp.float_, minimum=0.0, maximum=1.0)

        assert isinstance(spec.minimum, jnp.ndarray)
        assert isinstance(spec.maximum, jnp.ndarray)

        # Sanity check that jax compares correctly to a scalar for an empty shape.
        assert spec.minimum == 0.0
        assert spec.maximum == 1.0

        # Check that the spec doesn't fail its own input validation.
        _ = specs.BoundedArray(spec.shape, spec.dtype, spec.minimum, spec.maximum)

    def test_serialization(self) -> None:
        spec = specs.BoundedArray([1, 5], jnp.float32, -1, 1, "pickle_test")
        loaded_spec = pickle.loads(pickle.dumps(spec))
        assert isinstance(loaded_spec, spec.__class__)
        assert loaded_spec.dtype == spec.dtype
        assert loaded_spec.shape == spec.shape
        assert loaded_spec.name == spec.name
        assert jnp.all(loaded_spec.minimum == loaded_spec.minimum)
        assert jnp.all(loaded_spec.maximum == loaded_spec.maximum)

    @pytest.mark.parametrize(
        "arg_name, new_value",
        [
            ("shape", (2, 3)),
            ("dtype", jnp.int32),
            ("name", "something_else"),
            ("minimum", -2),
            ("maximum", 2),
        ],
    )
    def test_replace(self, arg_name: str, new_value: Any) -> None:
        old_spec = specs.BoundedArray([1, 5], jnp.float32, -1, 1, "test")
        new_spec = old_spec.replace(**{arg_name: new_value})
        assert old_spec is not new_spec
        assert getattr(new_spec, arg_name) == new_value
        for attr_name in {"shape", "dtype", "name", "minimum", "maximum"}.difference(
            [arg_name]
        ):
            assert getattr(new_spec, attr_name) == getattr(old_spec, attr_name)

    @pytest.mark.parametrize(
        "minimum, maximum",
        [
            (1.0, 0.0),
            ([0.0, 1.0], 0.0),
            (1.0, [0.0, 0.0]),
            ([0.0, 1.0], [0.0, 0.0]),
        ],
    )
    def test_error_if_minimum_greater_than_maximum(
        self,
        minimum: Union[chex.Numeric, Sequence],
        maximum: Union[chex.Numeric, Sequence],
    ) -> None:
        with pytest.raises(ValueError):
            specs.BoundedArray((2,), jnp.float32, minimum, maximum, "test")


class TestDiscreteArray:
    @pytest.mark.parametrize(
        "num_values",
        [0, -3],
    )
    def test_invalid_num_actions(self, num_values: int) -> None:
        with pytest.raises(ValueError):
            specs.DiscreteArray(num_values=num_values)

    @pytest.mark.parametrize(
        "dtype",
        [jnp.float32, float, str],
    )
    def test_dtype_not_integral(self, dtype: Union[jnp.dtype, type]) -> None:
        with pytest.raises(ValueError):
            specs.DiscreteArray(num_values=5, dtype=dtype)

    def test_repr(self) -> None:
        as_string = repr(specs.DiscreteArray(num_values=5))
        assert "5" in as_string

    def test_properties(self) -> None:
        num_values = 5
        spec = specs.DiscreteArray(num_values=5, dtype=jnp.int32)
        assert spec.minimum == 0
        assert spec.maximum == num_values - 1
        assert spec.dtype == jnp.int32
        assert spec.num_values == num_values

    def test_serialization(self) -> None:
        spec = specs.DiscreteArray(2, jnp.int32, "pickle_test")
        loaded_spec = pickle.loads(pickle.dumps(spec))
        assert isinstance(loaded_spec, spec.__class__)
        assert loaded_spec.dtype == spec.dtype
        assert loaded_spec.shape == spec.shape
        assert loaded_spec.name == spec.name
        assert jnp.all(loaded_spec.minimum == loaded_spec.minimum)
        assert jnp.all(loaded_spec.maximum == loaded_spec.maximum)
        assert loaded_spec.num_values == spec.num_values

    @pytest.mark.parametrize(
        "arg_name, new_value",
        [
            ("num_values", 4),
            ("dtype", jnp.int64),
            ("name", "something_else"),
        ],
    )
    def test_replace(self, arg_name: str, new_value: Any) -> None:
        old_spec = specs.DiscreteArray(2, jnp.int32, "test")
        new_spec = old_spec.replace(**{arg_name: new_value})
        assert old_spec is not new_spec
        assert getattr(new_spec, arg_name) == new_value
        for attr_name in {"num_values", "dtype", "name"}.difference([arg_name]):
            assert getattr(new_spec, attr_name) == getattr(old_spec, attr_name)


class TestJumanjiSpecsToDmEnvSpecs:
    def test_array(self) -> None:
        jumanji_spec = specs.Array((1, 2), jnp.int32)
        dm_env_spec = dm_env.specs.Array((1, 2), jnp.int32)
        converted_spec = specs.jumanji_specs_to_dm_env_specs(jumanji_spec)
        assert type(converted_spec) is type(dm_env_spec)
        assert converted_spec.shape == dm_env_spec.shape
        assert converted_spec.dtype == dm_env_spec.dtype
        assert converted_spec.name == dm_env_spec.name

    def test_bounded_array(self) -> None:
        jumanji_spec = specs.BoundedArray((1, 2), jnp.float32, minimum=0.0, maximum=1.0)
        dm_env_spec = dm_env.specs.BoundedArray(
            (1, 2), jnp.float32, minimum=0.0, maximum=1.0
        )
        converted_spec = specs.jumanji_specs_to_dm_env_specs(jumanji_spec)
        assert type(converted_spec) is type(dm_env_spec)
        assert converted_spec.shape == dm_env_spec.shape
        assert converted_spec.dtype == dm_env_spec.dtype
        assert converted_spec.name == dm_env_spec.name
        assert converted_spec.minimum == dm_env_spec.minimum
        assert converted_spec.maximum == dm_env_spec.maximum

    def test_discrete_array(self) -> None:
        jumanji_spec = specs.DiscreteArray(num_values=5, dtype=jnp.int32)
        dm_env_spec = dm_env.specs.DiscreteArray(num_values=5, dtype=jnp.int32)
        converted_spec = specs.jumanji_specs_to_dm_env_specs(jumanji_spec)
        assert type(converted_spec) is type(dm_env_spec)
        assert converted_spec.shape == dm_env_spec.shape
        assert converted_spec.dtype == dm_env_spec.dtype
        assert converted_spec.name == dm_env_spec.name
        assert converted_spec.minimum == dm_env_spec.minimum
        assert converted_spec.maximum == dm_env_spec.maximum
        assert converted_spec.num_values == dm_env_spec.num_values

    def test_spec(self, nested_spec_one_array: NestedSpecOneArray) -> None:
        with pytest.raises(ValueError):
            # Cannot work with nested specs.
            _ = specs.jumanji_specs_to_dm_env_specs(nested_spec_one_array)
