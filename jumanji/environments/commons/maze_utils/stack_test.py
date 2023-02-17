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

import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.commons.maze_utils.stack import (
    Stack,
    create_stack,
    empty_stack,
    stack_pop,
    stack_push,
)


class TestJaxStack:
    MAX_SIZE = 10
    ELEMENT_SIZE = 4

    @pytest.fixture
    def stack(self) -> Stack:
        return create_stack(self.MAX_SIZE, self.ELEMENT_SIZE)

    def test_create_stack(self, stack: Stack) -> None:
        assert stack.data.shape == (self.MAX_SIZE, self.ELEMENT_SIZE)
        assert jnp.all(stack.data == 0)

    def test_push(self, stack: Stack) -> None:
        a = jnp.array([1, 2, 3, 4])
        b = jnp.array([5, 6, 7, 8])

        push = jax.jit(stack_push)

        stack = push(stack, a)
        stack = push(stack, b)

        assert jnp.all(stack.data[0] == a)
        assert jnp.all(stack.data[1] == b)

    def test_pop(self, stack: Stack) -> None:
        a = jnp.array([1, 2, 3, 4])
        b = jnp.array([5, 6, 7, 8])

        stack = stack_push(stack, a)
        stack = stack_push(stack, b)

        pop = jax.jit(stack_pop)

        stack, element = pop(stack)
        assert jnp.all(element == b)
        stack, element = pop(stack)
        assert jnp.all(element == a)

    def test_empty(self, stack: Stack) -> None:
        assert empty_stack(stack)
        a = jnp.array([1, 2, 3, 4])
        stack = stack_push(stack, a)

        empty = jax.jit(empty_stack)
        assert not empty(stack)
        stack, _ = stack_pop(stack)
        assert empty(stack)
