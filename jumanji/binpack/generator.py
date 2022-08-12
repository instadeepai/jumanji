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
from typing import List, Tuple

import jax
import jax.numpy as jnp
import pandas as pd
from chex import PRNGKey

from jumanji.binpack.types import Container, Item, Location, State, empty_ems
from jumanji.tree_utils import tree_transpose

# 20 foot container: 5.898m long x 2.352m wide x 2.393m high
# Oftentimes people use different values (inner volume) : 5.870m long x 2.330m wide x 2.200m high
# real_container_volume = 1.103 * inner_container_volume
TWENTY_FOOT_DIMS = (5870, 2330, 2200)


def normalized_container(container_dims: Tuple[int, int, int]) -> Container:
    """Return a `Container` normalized by the tallest dimension.

    Args:
        container_dims: (length, width, height) tuple of integers corresponding to the dimensions
            of the container in millimeters.

    Returns:
        normalized container whose bottom-left corner is (0, 0, 0) and top-right corner is at max
            (1, 1, 1).

    """
    x2, y2, z2 = container_dims
    non_normalized_container = Container(x1=0, x2=x2, y1=0, y2=y2, z1=0, z2=z2).astype(
        jnp.float32
    )
    container: Container = jax.tree_map(
        lambda x: x / max(container_dims), non_normalized_container
    )
    return container


class InstanceGenerator(abc.ABC):
    """Defines the abstract `InstanceGenerator` base class. An `InstanceGenerator` is responsible
    for generating an instance when the environment is reset.
    """

    def __init__(self, max_num_items: int, max_num_ems: int):
        """Abstract class implementing `max_num_items` and `max_num_ems` properties.

        Args:
            max_num_items: maximum number of items the generator will ever generate when creating
                a new instance. This defines the shapes of arrays related to items in the
                environment state.
            max_num_ems: maximum number of ems the environment will handle. This defines the shape
                of the EMS buffer that is kept in the environment state.
        """
        self._max_num_items = max_num_items
        self._max_num_ems = max_num_ems

    @property
    def max_num_items(self) -> int:
        """Returns the maximum number of items the generator will ever create when resetting the
        environment."""
        return self._max_num_items

    @property
    def max_num_ems(self) -> int:
        """Returns the maximum number of ems the environment will handle. Any created ems that do
        not fit in the buffer will be ignored during the environment step."""
        return self._max_num_ems

    @abc.abstractmethod
    def __call__(self, key: PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: jax random key for any stochasticity used in the instance generation process.

        Returns:
            A BinPack State.
        """

    def generate_solution(self, key: PRNGKey) -> State:
        """Returns a state in which all items are placed to show a solution to the packing problem.
        The state corresponds to the end of an episode run with the optimal policy.

        Args:
            key: same jax random key used than when calling `__call__` to generate the same
                instance.

        Returns:
            A BinPack state whose items are all placed.
        """
        raise NotImplementedError


class SimpleInstanceGenerator(InstanceGenerator):
    """`InstanceGenerator` that can be used as an example. It deterministically outputs a single
    instance with 8 items that can be packed in a way to fully utilize a 20-ft container.
    """

    def __init__(self, max_num_ems: int = 40):
        """Instantiate a SimpleInstanceGenerator.

        Args:
            max_num_ems: maximum number of ems the environment will handle. This defines the shape
                of the EMS buffer that is kept in the environment state.
        """
        super().__init__(max_num_items=8, max_num_ems=max_num_ems)

    def __call__(self, key: PRNGKey) -> State:
        """Call method responsible for generating a new state. It returns a 20-ft container instance
        without any placed items.

        Args:
            key: jax random key for any stochasticity used in the generation process

        Returns:
            A BinPack State.
        """
        del key

        solution = self._generate_solved_instance()
        state = self._unpack_items(solution)

        return state

    def generate_solution(self, key: PRNGKey) -> State:
        """Returns a state in which all items are placed to show a solution to the packing problem.
        The state corresponds to the end of an episode that was run with the optimal policy.

        Args:
            key: same jax random key used than when calling `__call__` to generate the same
                instance.

        Returns:
            A BinPack state whose items are all placed.

        Example:
            ```python
            instance_generator = SimpleInstanceGenerator(...)
            env = BinPack(instance_generator, ...)
            key = jax.random.key(0)
            reset_state = instance_generator(key)
            env.render(reset_state)
            solution = instance_generator.generate_solution(key)
            env.render(solution)
            ```
        """
        del key
        solution = self._generate_solved_instance()
        return solution

    def _generate_solved_instance(self) -> State:
        """Generate the 20-ft instance with all items correctly packed."""
        container_dims = TWENTY_FOOT_DIMS

        container = normalized_container(container_dims)
        list_of_ems = [container] + [empty_ems() for _ in range(self.max_num_ems - 1)]
        ems = tree_transpose(list_of_ems)

        list_of_items = [
            Item(x_len=1717, y_len=1188, z_len=2200),
            Item(x_len=1794, y_len=1188, z_len=2200),
            Item(x_len=1809, y_len=423, z_len=2200),
            Item(x_len=1896, y_len=719, z_len=2200),
            Item(x_len=2359, y_len=488, z_len=2200),
            Item(x_len=2359, y_len=700, z_len=2200),
            Item(x_len=3974, y_len=719, z_len=2200),
            Item(x_len=4061, y_len=423, z_len=2200),
        ]

        items_non_normalized = tree_transpose(list_of_items)
        items = jax.tree_map(lambda x: x / max(container_dims), items_non_normalized)
        items_mask = jnp.ones(self.max_num_items, bool)

        sorted_ems_indexes = jnp.arange(0, self.max_num_ems, dtype=int)

        all_item_locations = Location(
            x=jnp.array(
                [
                    0.0,
                    0.0,
                    0.40182826,
                    0.0,
                    0.0,
                    0.70741236,
                    0.6918369,
                    0.6769486,
                    0.0,
                    0.0,
                ],
                float,
            ),
            y=jnp.array(
                [
                    0.0,
                    0.0,
                    0.40182826,
                    0.0,
                    0.0,
                    0.70741236,
                    0.6918369,
                    0.6769486,
                    0.0,
                    0.0,
                ],
                float,
            ),
            z=jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                float,
            ),
        )
        solution = State(
            container=container,
            ems=ems,
            ems_mask=jnp.zeros(self.max_num_ems, bool),
            items=items,
            items_mask=items_mask,
            items_placed=items_mask,
            items_location=all_item_locations,
            action_mask=None,
            sorted_ems_indexes=sorted_ems_indexes,
        )

        return solution

    def _unpack_items(self, state: State) -> State:
        """Unpack any items that were packed in the given state. Reset to how the state was at the
        beginning of the episode.
        """
        state.ems_mask = jnp.zeros(self.max_num_ems, bool).at[0].set(True)
        state.items_placed = jnp.zeros(self.max_num_items, bool)
        state.items_location = Location(
            x=jnp.zeros(self.max_num_items, float),
            y=jnp.zeros(self.max_num_items, float),
            z=jnp.zeros(self.max_num_items, float),
        )
        return state


class CSVInstanceGenerator(InstanceGenerator):
    """`InstanceGenerator` that parses a CSV file to do active search on a single instance. It
    always resets to the same instance defined by the CSV file. The generator can handle any
    container dimensions but assumes a 20-ft container by default.

    The CSV file is expected to have the following columns:
    - Product_Name
    - Length
    - Width
    - Height
    - Quantity
    - Stackable

    Example of such a CSV file:
        Product_Name,Length,Width,Height,Quantity,Stackable
        shape_1,1080,760,300,5,2
        shape_2,1100,430,250,3,1
    """

    def __init__(
        self,
        csv_path: str,
        max_num_ems: int,
        container_dims: Tuple[int, int, int] = TWENTY_FOOT_DIMS,
    ):
        """Instantiate a CSVInstanceGenerator that generates the same instance (active search)
        defined by a CSV file.

        Args:
            csv_path: path to the CSV file defining the instance to reset to.
            max_num_ems: maximum number of ems the environment will handle. This defines the shape
                of the EMS buffer that is kept in the environment state.
            container_dims: (length, width, height) tuple of integers corresponding to the
                dimensions of the container in millimeters. By default, assume a 20-ft container.
        """
        self.container_dims = container_dims
        self.instance_from_csv = self._parse_csv_file(csv_path, max_num_ems)
        max_num_items = self.instance_from_csv.items_mask.shape[0]
        super().__init__(max_num_items, max_num_ems)

    def __call__(self, key: PRNGKey) -> State:
        """Call method responsible for generating a new state.

        Args:
            key: key is not used as the instance is deterministically generated from CSV file.

        Returns:
            A BinPack State that corresponds to the instance defined by the CSV file.
        """
        del key
        return self.instance_from_csv

    def _parse_csv_file(self, csv_path: str, max_num_ems: int) -> State:
        """Create an instance by parsing a CSV file.

        Args:
            csv_path: path to the CSV file to parse that defines the instance to reset to.
            max_num_ems: maximum number of ems the environment will handle. This defines the shape
                of the EMS buffer that is kept in the environment state.

        Returns:
            BinPack state that contains the instance defined in the CSV file.
        """
        container = normalized_container(self.container_dims)

        # Initialize the EMS
        list_of_ems = [container] + (max_num_ems - 1) * [empty_ems()]
        ems = tree_transpose(list_of_ems)
        ems_mask = jnp.zeros(max_num_ems, bool).at[0].set(True)

        # Parse the CSV file to generate the items
        data_frame = pd.read_csv(csv_path)
        list_of_items = self._generate_list_of_items(data_frame)
        items = tree_transpose(list_of_items)

        # Initialize items mask and location
        num_items = len(list_of_items)
        items_mask = jnp.ones(num_items, bool)
        items_placed = jnp.zeros(num_items, bool)
        items_location = Location(
            x=jnp.zeros(num_items, float),
            y=jnp.zeros(num_items, float),
            z=jnp.zeros(num_items, float),
        )

        sorted_ems_indexes = jnp.arange(0, max_num_ems, dtype=int)
        reset_state = State(
            container=container,
            ems=ems,
            ems_mask=ems_mask,
            items=items,
            items_mask=items_mask,
            items_placed=items_placed,
            items_location=items_location,
            action_mask=None,
            sorted_ems_indexes=sorted_ems_indexes,
        )

        return reset_state

    def _generate_list_of_items(self, data_frame: pd.DataFrame) -> List[Item]:
        """Generate the list of items from a Pandas DataFrame.

        Args:
            data_frame: Pandas DataFrame describing the items for the corresponding instance.

        Returns:
            List of Item flattened so that identical items (quantity > 1) are copied according to
                their quantity.
        """
        max_size = max(self.container_dims)
        list_of_items = []
        for _, (_, x_len, y_len, z_len, quantity, _) in data_frame.iterrows():
            identical_items = quantity * [
                Item(
                    x_len=jnp.array(x_len / max_size, float),
                    y_len=jnp.array(y_len / max_size, float),
                    z_len=jnp.array(z_len / max_size, float),
                )
            ]
            list_of_items.extend(identical_items)
        return list_of_items


def save_instance_to_csv(
    state: State, path: str, container_dims: Tuple[int, int, int] = TWENTY_FOOT_DIMS
) -> pd.DataFrame:
    """Save an instance to a CSV file. Items are supposed to be normalized by the container_dims
    dimensions and have length, width and height between 0 and 1. The conversion to CSV
    will convert the item dimensions back to millimeters.

    Args:
        state: BinPack State.
        path: path to save the CSV file.
        container_dims: (length, width, height) tuple of integers corresponding to the
            dimensions of the container in millimeters. By default, assume a 20-ft container.

    Returns:
        A Pandas dataframe that contains the instance data saved to CSV.

    Example:
        Product_Name,Length,Width,Height,Quantity,Stackable
        shape_1,1080,760,300,5,2
        shape_2,1100,430,250,3,1
    """
    max_size = max(container_dims)

    # Rescale items to their real size and set to 0 empty items
    rescaled_items = jax.tree_map(
        lambda x: round(x * max_size * state.items_mask), state.items
    )
    data_frame = pd.DataFrame(rescaled_items._asdict())
    data_frame.columns = ["Length", "Width", "Height"]

    # Remove empty items
    data_frame = data_frame[
        (data_frame["Length"] > 0)
        & (data_frame["Width"] > 0)
        & (data_frame["Height"] > 0)
    ]

    # Add missing fields
    data_frame["Quantity"] = 1
    data_frame["Stackable"] = 1

    # Aggregate same items together
    aggregation_functions = {
        "Length": "first",
        "Width": "first",
        "Height": "first",
        "Quantity": "sum",
        "Stackable": "first",
    }
    data_frame = data_frame.groupby(["Length", "Width", "Height"]).aggregate(
        aggregation_functions
    )
    data_frame = data_frame.sort_values("Quantity", ascending=False)

    # Make Product_Name column
    data_frame.index = [f"shape_{i}" for i in range(1, len(data_frame.Length) + 1)]
    data_frame = data_frame.reset_index().rename(columns={"index": "Product_Name"})
    data_frame.to_csv(path, index=False)
    return data_frame
