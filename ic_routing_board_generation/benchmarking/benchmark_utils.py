import os
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

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
