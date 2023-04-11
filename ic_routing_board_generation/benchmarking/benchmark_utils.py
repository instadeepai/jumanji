import pickle
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters
from ic_routing_board_generation.interface.board_generator_interface import \
    BoardName, BoardGenerator


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


def board_generation_params_from_grid_params(
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
    directory = directory_string_from_benchamrk_experiement(benchmark_experiment)
    path = directory.glob('**/*')
    return [file_path.name for file_path in path if file_path.is_file()]

def directory_string_from_benchamrk_experiement(benchmark_experiment: str) -> Path:
    dir_string = Path(__file__).parent.parent.parent
    folder = f"ic_experiments/benchmarks/{benchmark_experiment}/"
    return dir_string / folder


def generate_n_boards(
    board_parameters: BoardGenerationParameters,
    number_of_boards: int,
):
    # TODO (Marta): add exception of all board_gen_parameters are not the same (with the exception of board_type
    board_list = []
    board_class = BoardGenerator.get_board_generator(board_parameters.generator_type)
    board_generator = board_class(
        rows=board_parameters.rows, cols=board_parameters.columns,
        num_agents=board_parameters.number_of_wires,
    )
    for _ in range(number_of_boards):
        board = None
        none_counter = 0
        while board is None:
            board_generator = board_class(
                rows=board_parameters.rows, cols=board_parameters.columns,
                num_agents=board_parameters.number_of_wires,
            )
            board = board_generator.return_solved_board()
            none_counter += 1
            if none_counter == 100:
                raise ValueError("Failed to generate board 100 times")
        board_list.append(board)

    return board_list
