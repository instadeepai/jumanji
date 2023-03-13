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

"""Define a stack data structure which can be used with Jax.

To be usable with Jax transformations, data structures must have fixed shape.
A stack can be represented by a 2D array, each row containing the flatten representation of an
element. For example, in a stack of chambers (as in the maze generation), each row should
contain 4 digits: x0, y0, width and height.
We also need an upper bound to the number of elements the stack can contain, so that we always
have a fixed number of row.
To create the stack we need two parameters: the size of one element, and the maximum number
of elements.

For example, here is an empty stack containing elements of size 4 with at most 5 elements:

[[. . . .]
[. . . .]
[. . . .]
[. . . .]
[. . . .]]

Originally the stack is empty, data only contains padding. Elements can be pushed on the stack.
Say an element `[a,b,c,d]` is pushed on the stack for example:

[[a b c d]
[. . . .]
[. . . .]
[. . . .]
[. . . .]]

In this 2D array, how do we differentiate between the first row, which actually contains an element,
and the other rows, which only contain padding ?
An `insertion_index` can be used, which contains the index at which the next element should
be inserted. All rows up to this index are elements of the stack, all rows after contain padding.

[[a b c d]
[. . . .] <- insertion_index # everything from this row is padding and should be ignored
[. . . .]
[. . . .]
[. . . .]]

"""
from typing import NamedTuple, Tuple

import chex
import jax.numpy as jnp


class Stack(NamedTuple):
    """Define a stack usable with Jax transformations.

    - data: array of fixed-shape, each row up to insertion_index containing an element of the stack.
        Rows after insertion_index should be ignored, they only contain padding to make sure data
        is of fixed shape and can be used with Jax transformations.
        The width of the data is the number of features in an element, the height is the maximum
        number of elements the stack can contain.
    - insertion_index: the index of the row at which to insert the next element in data. Should be
        0 for an empty stack.
    """

    data: chex.Array
    insertion_index: int


def create_stack(max_size: int, num_features: int) -> Stack:
    """Create an empty stack.

    Args:
        max_size: maximum number of elements the stack can contain.
        num_features: number of features in an element.

    Returns:
        stack: the created stack.
    """
    return Stack(jnp.zeros((max_size, num_features), dtype=int), 0)


def stack_push(stack: Stack, element: chex.Array) -> Stack:
    """Push an element on top of the stack.

    Args:
        stack: the stack on which to push element.
        element: the element to push on the stack.

    Returns:
        stack: the stack containing the new element.
    """
    return Stack(
        stack.data.at[stack.insertion_index].set(element),
        stack.insertion_index + 1,
    )


def stack_pop(stack: Stack) -> Tuple[Stack, chex.Array]:
    """Pop the last element from the stack.

    Args:
        stack: the stack from which to pop the last element.

    Returns:
        stack: the stack without the last element.
        element: the last element from the stack.
    """
    last_element_idx = stack.insertion_index - 1
    element = stack.data[last_element_idx]
    stack = Stack(
        stack.data,
        last_element_idx,
    )
    return stack, element


def empty_stack(stack: Stack) -> bool:
    """Check if a stack is empty.

    Args:
        stack: the stack to check.

    Returns:
        Boolean stating whether the stack is empty.
    """
    return stack.insertion_index == 0
