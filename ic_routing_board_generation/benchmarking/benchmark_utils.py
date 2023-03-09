import math
import os
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters
from ic_routing_board_generation.interface.board_generator_interface import \
    BoardName


def load_pickle(filename: str):
    with open(filename, "rb") as file:
        benchmark = pickle.load(file)

def make_benchmark_folder(with_time: bool = True):
    path = Path(__file__).parent.parent.parent
    today = date.today().strftime("%Y%m%d")
    time_now = datetime.now().strftime("%H_%M") if with_time else ""
    experiment_folder = f"ic_experiments/benchmarks/{today}_benchmark_{time_now}/"
    new_path = path / experiment_folder
    new_path.mkdir(exist_ok=True)
    return path / experiment_folder


def generate_board_generation_params(
    grid_parameters: List[Tuple[int, int, int]],
) -> List[BoardGenerationParameters]:
    benchmarks_list = []
    for board_generator in BoardName:
        for parameters in grid_parameters:
            benchmark = BoardGenerationParameters(
                rows=parameters[0], columns=parameters[0],
                number_of_wires=parameters[2],
                generator_type=board_generator,
            )
            benchmarks_list.append(benchmark)
    return benchmarks_list


def files_list_from_benchmark_experiment(
        benchmark_experiment: str) -> List[str]:
    directory = return_directory_string(benchmark_experiment)
    path = directory.glob('**/*')
    return [file_path.name for file_path in path if file_path.is_file()]


def return_directory_string(benchmark_experiment: str) -> Path:
    dir_string = Path(__file__).parent.parent.parent
    folder = f"ic_experiments/benchmarks/{benchmark_experiment}/"
    return dir_string / folder


def plot_heatmap(scores: np.ndarray):
    # TODO (Marta): Add saving capability
    fig, ax = plt.subplots()
    im = ax.imshow(scores, cmap='Purples', interpolation='nearest')
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def plot_comparison_heatmap(
    list_of_scores: List[np.ndarray],
    list_of_titles: List[str],
    num_agents: int,
    number_of_boards_averaged: int,
):

    n_rows = max(math.ceil(len(list_of_scores) / 3), 1)
    print(n_rows)
    n_columns = 3
    fig = plt.figure(figsize=(6, 2 * n_rows + 0.5))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(n_rows, n_columns),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )
    plt.suptitle(f"Scores per Cell Averaged on {number_of_boards_averaged} Boards with {num_agents} wires ", fontsize=12, y=0.98)

    for i, ax in enumerate(grid):

        if i >= len(list_of_scores):
            fig.delaxes(ax)
        else:
            ax.set_title(list_of_titles[i])
            im = ax.imshow(list_of_scores[i], cmap='Purples')
            ax.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

    cbar = ax.cax.colorbar(im)
    cbar = grid.cbar_axes[0].colorbar(im)
    plt.tight_layout()
    plt.show()
