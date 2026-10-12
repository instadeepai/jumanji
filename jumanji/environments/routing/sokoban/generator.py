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
import os
import zipfile
from os import listdir
from os.path import isfile, join
from typing import List, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import requests

from jumanji.environments.routing.sokoban.constants import (
    AGENT,
    BOX,
    EMPTY,
    GRID_SIZE,
    TARGET,
    WALL,
)
from jumanji.environments.routing.sokoban.types import State

# Maps each character of the Boxoban text format to its (fixed, variable) encoding.
CELL_MAPPING = {
    "#": (WALL, EMPTY),
    ".": (TARGET, EMPTY),
    "@": (EMPTY, AGENT),
    "$": (EMPTY, BOX),
    " ": (EMPTY, EMPTY),
}

BOXOBAN_URL = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
BOXOBAN_SPLITS = {
    "unfiltered": ("train", "valid", "test"),
    "medium": ("train", "valid"),
    "hard": (None,),
}


class Generator(abc.ABC):
    def __init__(
        self,
    ) -> None:
        """ """

        self._fixed_grids: chex.Array
        self._variable_grids: chex.Array

    @abc.abstractmethod
    def __call__(self, rng_key: chex.PRNGKey) -> State:
        """Generate a problem instance.

        Args:
            key: the Jax random number generation key.

        Returns:
            state: the generated problem instance.
        """

    def get_agent_coordinates(self, grid: chex.Array) -> chex.Array:
        """Extracts the coordinates of the agent from a given grid with the
        assumption there is only one agent in the grid.

        Args:
            grid: Array (uint8) of shape (num_rows, num_cols)

        Returns:
            location: (int32) of shape (2,)
        """

        coordinates = jnp.where(grid == AGENT, size=1)

        x_coord = jnp.squeeze(coordinates[0])
        y_coord = jnp.squeeze(coordinates[1])

        return jnp.array([x_coord, y_coord])


class DeepMindGenerator(Generator):
    """Instance generator that generates a random problem from the DeepMind
    Boxoban dataset, a popular dataset for comparing Reinforcement Learning
    algorithms and Planning Algorithms. The dataset has unfiltered, medium and
    hard versions. The unfiltered dataset contains train, test and valid
    splits. The medium has train and valid splits available. And the hard set
    contains just a small number of problems and no splits. The problems are
    all guaranteed to be solvable.

    The dataset is downloaded from GitHub on first use, parsed once and cached
    as a `.npy` file in `~/.cache/sokoban_dataset`.
    """

    def __init__(
        self,
        difficulty: str,
        split: Optional[str],
        proportion_of_files: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """Instantiates a `DeepMindGenerator`.

        Args:
            difficulty: one of "unfiltered", "medium" or "hard".
            split: "train", "valid" or "test" for "unfiltered", "train" or
                "valid" for "medium". Ignored for "hard".
            proportion_of_files: float between (0,1) for the proportion of
                problems to use in the dataset.
            verbose: whether to print download progress.
        """
        if difficulty not in BOXOBAN_SPLITS:
            raise ValueError(
                f"Invalid Boxoban difficulty {difficulty!r}, "
                f"expected one of {list(BOXOBAN_SPLITS)}."
            )
        if difficulty == "hard":
            split = None
        elif split not in BOXOBAN_SPLITS[difficulty]:
            raise ValueError(
                f"Invalid split {split!r} for Boxoban difficulty {difficulty!r}, "
                f"expected one of {list(BOXOBAN_SPLITS[difficulty])}."
            )

        self.difficulty = difficulty
        self.split = split
        self.verbose = verbose
        self.proportion_of_files = proportion_of_files

        # Set the cache path to user's home directory's .cache sub-directory
        self.cache_path = os.path.join(os.path.expanduser("~"), ".cache", "sokoban_dataset")

        dataset = self._load_dataset()

        # Convert to jax arrays and resize using proportion_of_files
        length = int(proportion_of_files * dataset.shape[0])
        self._fixed_grids = jnp.asarray(dataset[:length, ..., 0], jnp.uint8)
        self._variable_grids = jnp.asarray(dataset[:length, ..., 1], jnp.uint8)

    def __call__(self, rng_key: chex.PRNGKey) -> State:
        """Generate a random Boxoban problem from the Deepmind dataset.

        Args:
            rng_key: the Jax random number generation key.

        Returns:
            fixed_grid: Array (uint8) shape (num_rows, num_cols) the fixed
            components of the problem.
            variable_grid: Array (uint8) shape (num_rows, num_cols) the
            variable components of the problem.
        """

        key, idx_key = jax.random.split(rng_key)
        idx = jax.random.randint(idx_key, shape=(), minval=0, maxval=self._fixed_grids.shape[0])
        fixed_grid = self._fixed_grids.take(idx, axis=0)
        variable_grid = self._variable_grids.take(idx, axis=0)

        initial_agent_location = self.get_agent_coordinates(variable_grid)

        state = State(
            key=key,
            fixed_grid=fixed_grid,
            variable_grid=variable_grid,
            agent_location=initial_agent_location,
            step_count=jnp.array(0, jnp.int32),
        )

        return state

    def _load_dataset(self) -> np.ndarray:
        """Loads the parsed dataset from the cache, downloading and parsing the
        text files on first use.

        Returns:
            dataset: Array (uint8) shape (dataset_size, num_rows, num_cols, 2)
            with the fixed grids in channel 0 and the variable grids in channel 1.
        """
        dataset_name = self.difficulty if self.split is None else f"{self.difficulty}-{self.split}"
        dataset_file = os.path.join(self.cache_path, f"{dataset_name}.npy")
        if os.path.exists(dataset_file):
            return np.load(dataset_file)

        self._download_data()
        data_dir = os.path.join(self.cache_path, "boxoban-levels-master", self.difficulty)
        if self.split is not None:
            data_dir = os.path.join(data_dir, self.split)
        dataset = self._generate_dataset(data_dir)

        # Write to a temporary file first so an interrupted run never leaves a
        # truncated cache behind.
        tmp_file = f"{dataset_file}.{os.getpid()}.tmp"
        with open(tmp_file, "wb") as f:
            np.save(f, dataset)
        os.replace(tmp_file, dataset_file)
        return dataset

    @staticmethod
    def _generate_dataset(data_dir: str) -> np.ndarray:
        """Parses the text files of a Boxoban split into a single array.

        Args:
            data_dir: directory containing the text files of the split.

        Returns:
            dataset: Array (uint8) shape (dataset_size, num_rows, num_cols, 2)
            with the fixed grids in channel 0 and the variable grids in channel 1.
        """
        all_files = sorted(f for f in listdir(data_dir) if isfile(join(data_dir, f)))

        # Every level is GRID_SIZE rows that all start with a wall, so keeping
        # those rows in order gives the levels back to back.
        rows: List[str] = []
        for file in all_files:
            with open(join(data_dir, file), "r") as sf:
                rows.extend(line.rstrip("\r\n") for line in sf if line.startswith("#"))

        chars = np.frombuffer("".join(rows).encode("ascii"), dtype=np.uint8)
        chars = chars.reshape(-1, GRID_SIZE, GRID_SIZE)

        lookup = np.full((256, 2), 255, dtype=np.uint8)
        for cell, encoding in CELL_MAPPING.items():
            lookup[ord(cell)] = encoding
        dataset = lookup[chars]
        if (dataset == 255).any():
            raise ValueError(f"Unexpected character in Boxoban levels in {data_dir}.")
        return dataset

    def _download_data(self) -> None:
        """Downloads the deepmind boxoban dataset from github into text files"""

        # Check if the cache directory exists, if not, create it
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # Check if the dataset is already downloaded in the cache
        dataset_path = os.path.join(self.cache_path, "boxoban-levels-master")
        if not os.path.exists(dataset_path):
            if self.verbose:
                print("Boxoban: Pregenerated levels not downloaded.")
                print('Starting download from "{}"'.format(BOXOBAN_URL))

            response = requests.get(BOXOBAN_URL, stream=True, timeout=60)

            if response.status_code != 200:
                raise Exception("Could not download levels")

            path_to_zip_file = os.path.join(self.cache_path, "boxoban_levels-master.zip")
            with open(path_to_zip_file, "wb") as handle:
                for data in response.iter_content(chunk_size=1 << 20):
                    handle.write(data)

            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                zip_ref.extractall(self.cache_path)


class ToyGenerator(Generator):
    def __call__(
        self,
        rng_key: chex.PRNGKey,
    ) -> State:
        """Generate a random Boxoban problem from the toy 2 problem dataset.

        Args:
            rng_key: the Jax random number generation key.

        Returns:
            fixed_grid: Array (uint8) shape (num_rows, num_cols) the fixed
            components of the problem.
            variable_grid: Array (uint8) shape (num_rows, num_cols) the
            variable components of the problem.
        """

        key, idx_key = jax.random.split(rng_key)

        level1 = [
            "##########",
            "# @      #",
            "# $    . #",
            "#  $# .  #",
            "#  .#$  # ",
            "# . # $ # ",
            "#        #",
            "##########",
            "##########",
            "##########",
        ]

        level2 = [
            "##########",
            "#        #",
            "#$ #   . #",
            "# # $ # .#",
            "#  .# $  #",
            "# @ # . $#",
            "#        #",
            "##########",
            "##########",
            "##########",
        ]

        game1_fixed, game1_variable = convert_level_to_array(level1)
        game2_fixed, game2_variable = convert_level_to_array(level2)

        games_fixed = jnp.stack([game1_fixed, game2_fixed])
        games_variable = jnp.stack([game1_variable, game2_variable])

        game_index = jax.random.randint(
            key=idx_key,
            shape=(),
            minval=0,
            maxval=games_fixed.shape[0],
        )

        initial_agent_location = self.get_agent_coordinates(games_variable[game_index])

        state = State(
            key=key,
            fixed_grid=games_fixed[game_index],
            variable_grid=games_variable[game_index],
            agent_location=initial_agent_location,
            step_count=jnp.array(0, jnp.int32),
        )

        return state


class SimpleSolveGenerator(Generator):
    def __call__(
        self,
        key: chex.PRNGKey,
    ) -> State:
        """Generate a trivial Boxoban problem.

        Args:
            key: the Jax random number generation key.

        Returns:
            fixed_grid: Array (uint8) shape (num_rows, num_cols) the fixed
            components of the problem.
            variable_grid: Array (uint8) shape (num_rows, num_cols) the
            variable components of the problem.
        """
        level1 = [
            "##########",
            "#       ##",
            "# ....   #",
            "# $$$$  ##",
            "# @    # #",
            "#   #   # ",
            "#        #",
            "##########",
            "##########",
            "##########",
        ]

        game_fixed, game_variable = convert_level_to_array(level1)

        initial_agent_location = self.get_agent_coordinates(game_variable)

        state = State(
            key=key,
            fixed_grid=game_fixed,
            variable_grid=game_variable,
            agent_location=initial_agent_location,
            step_count=jnp.array(0, jnp.int32),
        )

        return state


def convert_level_to_array(level: List[str]) -> Tuple[chex.Array, chex.Array]:
    """Converts text representation of levels to a tuple of Jax arrays
    representing the fixed elements of the Boxoban problem and the variable
    elements

    Args:
        level: List of str representing a boxoban level.

    Returns:
        fixed_grid: Array (uint8) shape (num_rows, num_cols)
        the fixed components of the problem.
        variable_grid: Array (uint8) shape (num_rows,
        num_cols) the variable components of the problem.
    """

    fixed = [[CELL_MAPPING[cell][0] for cell in row] for row in level]
    variable = [[CELL_MAPPING[cell][1] for cell in row] for row in level]

    return jnp.array(fixed, jnp.uint8), jnp.array(variable, jnp.uint8)
