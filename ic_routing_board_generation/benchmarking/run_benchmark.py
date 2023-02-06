from typing import List

from ic_routing_board_generation.benchmarking.benchmark_utils import \
    generate_board_generation_params, files_list_from_benchmark_experiment, \
    return_directory_string
from ic_routing_board_generation.benchmarking.mw_benchmark import \
    BasicBenchmark
from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters


def run_benchmark_with_simulation(
    benchmarks_list: List[BoardGenerationParameters],
    save_plots: bool = False,
    save_simulation_data: bool = False,
    number_of_runs: int = 1000,
):
    benchmark = BasicBenchmark.from_simulation(
        benchmark_parameters_list=benchmarks_list,
        number_of_runs=number_of_runs,
        save_outputs=save_simulation_data,
    )
    benchmark.plot_all(save_outputs=save_plots)

def run_benchmark_from_file(
    files_for_benchmark: List[str],
    directory_string: str,
    save_plots: bool = False,
):
    benchmark = BasicBenchmark.from_file(
        file_name_parameters=files_for_benchmark,
        directory_string=directory_string,
    )
    benchmark.plot_all(save_outputs=save_plots)


if __name__ == '__main__':
    # set to True if you want to simulate the board, False if you want to run from file
    simulation = False

    if simulation:

        ######### Change these parameters are required
        grid_params = [(8, 8, 3), (12, 12, 4)]
        save_plots = True  # Change this to False if you want to just see the plots without saving
        save_simulation_data = True
        number_of_boards = 5
        #########

        benchmarks_list = generate_board_generation_params(grid_params)
        run_benchmark_with_simulation(
            benchmarks_list=benchmarks_list,
            save_plots=save_plots,
            save_simulation_data=save_simulation_data,
            number_of_runs=number_of_boards, # number of boards for simulation
        )
    else:
        ######### Change these parameters are required
        folder_name = "20230206_benchmark_20_16" # this must be a folder under ic/experiments/benchmarks
        save_plots = True
        # Option 1: get all files from folder
        all_files = files_list_from_benchmark_experiment(folder_name)

        # Option 2: Provide board generation parameters
        # TODO (MW): add example
        #########

        directory_string = str(return_directory_string(folder_name)) + "/"
        run_benchmark_from_file(
            files_for_benchmark=all_files,
            directory_string=directory_string,
            save_plots=save_plots,
        )
