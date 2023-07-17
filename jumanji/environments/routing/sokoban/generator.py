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
from typing import List, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm


class Generator(abc.ABC):
    """Defines the abstract `Generator` base class. A `Generator` is responsible
    for generating a problem instance when the environment is reset.
    """

    @abc.abstractmethod
    def __call__(self, rng_key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
        """Generate a problem instance.

        Args:
            key: the Jax random number generation key.

        Returns:
            state: the generated problem instance.
        """


class DeepMindGenerator(Generator):
    """Instance generator that generates a random problem from the DeepMind
    Boxoban dataset a popular dataset for comparing Reinforcement Learning
    algorithms and Planning Algorithms. The dataset has unfiltered, medium and
    hard versions. The unfiltered dataset contain train, test and valid
    splits. The Medium has train and valid splits available. And the hard set
    contains just a small number of problems. The problems are all guaranteed
    to be solvable.
    """

    def __init__(
        self,
        difficulty: str,
        split: str,
        proportion_of_files: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """Instantiates a `DeepMindGenerator`.

        Args:
            difficulty: which difficulty subset to use
            (hard,medium,unfiltered).
            split: which part of the dataset to use (train,test,valid).
            proportion_of_files: float between (0,1) for the proportion of
            files to use in the dataset .
            verbose: bool activating verbose mode.
        """

        self.difficulty = difficulty
        self.verbose = verbose
        self.proportion_of_files = proportion_of_files
        # Downloads data if not already downloaded
        self.cache_path = ".sokoban_cache"
        self._download_data()
        self.train_data_dir = os.path.join(
            self.cache_path, "boxoban-levels-master", self.difficulty
        )

        if self.difficulty in ["unfiltered", "medium", "hard"]:
            self.train_data_dir = os.path.join(
                self.train_data_dir,
                split,
            )
        # Generates the dataset of sokoban levels
        self._fixed_grids, self._variable_grids = self._generate_dataset()

    def __call__(self, rng_key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
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
        idx = jax.random.randint(
            idx_key, shape=(), minval=0, maxval=self._fixed_grids.shape[0]
        )
        fixed_grid = self._fixed_grids.take(idx, axis=0)
        variable_grid = self._variable_grids.take(idx, axis=0)

        return fixed_grid, variable_grid

    def _generate_dataset(
        self,
    ) -> Tuple[chex.Array, chex.Array]:
        """Parses the text files to generate a jax arrays (fixed and variable
        grids representing the Boxoban dataset

        Returns:
            fixed_grid: Array (uint8) shape (dataset_size, num_rows, num_cols)
            the fixed components of the problem.
            variable_grid: Array (uint8) shape (dataset_size, num_rows,
            num_cols) the variable components of the problem.
        """

        all_files = [
            f
            for f in listdir(self.train_data_dir)
            if isfile(join(self.train_data_dir, f))
        ]
        # Only keep a few files if specified
        all_files = all_files[: int(self.proportion_of_files * len(all_files))]

        fixed_grids_list: List[chex.Array] = []
        variable_grids_list: List[chex.Array] = []
        for file in all_files:
            source_file = join(self.train_data_dir, file)
            current_map: List[str] = []
            # parses a game file containing multiple games
            with open(source_file, "r") as sf:
                for line in sf.readlines():
                    if ";" in line and current_map:
                        fixed_grid, variable_grid = convert_level_to_array(current_map)
                        fixed_grids_list.append(fixed_grid)
                        variable_grids_list.append(variable_grid)
                        current_map = []
                    if "#" == line[0]:
                        current_map.append(line.strip())
        fixed_grids = jnp.asarray(fixed_grids_list, jnp.uint8)
        variable_grids = jnp.asarray(variable_grids_list, jnp.uint8)

        return fixed_grids, variable_grids

    def _download_data(self) -> None:
        """Downloads the deepmind boxoban dataset from github into text
        files"""

        if not os.path.exists(self.cache_path):
            url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
            if self.verbose:
                print("Boxoban: Pregenerated levels not downloaded.")
                print('Starting download from "{}"'.format(url))

            response = requests.get(url, stream=True)

            if response.status_code != 200:
                raise Exception("Could not download levels")

            os.makedirs(self.cache_path)
            path_to_zip_file = os.path.join(
                self.cache_path, "boxoban_levels-master.zip"
            )
            with open(path_to_zip_file, "wb") as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)

            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                zip_ref.extractall(self.cache_path)


class HuggingFaceDeepMindGenerator(Generator):
    """Instance generator that generates a random problem from the DeepMind
    Boxoban dataset a popular dataset for comparing Reinforcement Learning
    algorithms and Planning Algorithms. The dataset has unfiltered, medium and
    hard versions. The unfiltered dataset contain train, test and valid
    splits. The Medium has train and valid splits available. And the hard set
    contains just a small number of problems. The problems are all guaranteed
    to be solvable.
    """

    def __init__(
        self,
        dataset_name: str,
        proportion_of_files: float = 1.0,
    ) -> None:
        """Instantiates a `DeepMindGenerator`.

        Args:
            dataset_name: the name of the dataset to use. Choices are:
                - unfiltered-train,
                - unfiltered-valid,
                - unfiltered-test,
                - medium-train,
                - medium-test,
                - hard.
            proportion_of_files: float between (0,1) for the proportion of
            files to use in the dataset .
        """

        self.dataset_name = dataset_name
        self.proportion_of_files = proportion_of_files

        dataset_file = hf_hub_download(
            repo_id="InstaDeepAI/boxoban-levels", filename=f"{dataset_name}.npy"
        )
        with open(dataset_file, "rb") as f:
            dataset = np.load(f)

        # Convert to jax arrays and resize using proportion_of_files
        length = int(proportion_of_files * dataset.shape[0])
        self._fixed_grids = jnp.asarray(dataset[:length, ..., 0], jnp.uint8)
        self._variable_grids = jnp.asarray(dataset[:length, ..., 1], jnp.uint8)

    def __call__(self, rng_key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
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
        idx = jax.random.randint(
            idx_key, shape=(), minval=0, maxval=self._fixed_grids.shape[0]
        )
        fixed_grid = self._fixed_grids.take(idx, axis=0)
        variable_grid = self._variable_grids.take(idx, axis=0)

        return fixed_grid, variable_grid


class ToyGenerator(Generator):
    def __call__(
        self,
        rng_key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array]:
        """Generate a random Boxoban problem from the toy 2 problem dataset.

        Args:
            rng_key: the Jax random number generation key.

        Returns:
            fixed_grid: Array (uint8) shape (num_rows, num_cols) the fixed
            components of the problem.
            variable_grid: Array (uint8) shape (num_rows, num_cols) the
            variable components of the problem.
        """

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
            key=rng_key,
            shape=(),
            minval=0,
            maxval=games_fixed.shape[0],
        )

        return games_fixed[game_index], games_variable[game_index]


class SimpleSolveGenerator(Generator):
    def __call__(
        self,
        rng_key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array]:
        """Generate a trivial Boxoban problem.

        Args:
            rng_key: the Jax random number generation key.

        Returns:
            fixed_grid: Array (uint8) shape (num_rows, num_cols) the fixed
            components of the problem.
            variable_grid: Array (uint8) shape (num_rows, num_cols) the
            variable components of the problem.
        """
        del rng_key

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

        return game_fixed, game_variable


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

    # Define the mappings
    mapping = {
        "#": (1, 0),
        ".": (2, 0),
        "@": (0, 3),
        "$": (0, 4),
        " ": (0, 0),  # empty cell
    }

    fixed = [[mapping[cell][0] for cell in row] for row in level]
    variable = [[mapping[cell][1] for cell in row] for row in level]

    return jnp.array(fixed, jnp.uint8), jnp.array(variable, jnp.uint8)
