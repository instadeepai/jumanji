import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Iterable, Union
import pickle

from matplotlib import pyplot as plt
import jax.numpy as jnp

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters, BenchmarkData
from ic_routing_board_generation.benchmarking.benchmark_utils import \
    make_benchmark_folder
from ic_routing_board_generation.ic_routing.route import Route
from ic_routing_board_generation.interface.board_generator_interface import \
    BoardGenerators


class BasicBenchmark:
    def __init__(
        self,
        benchmark_data: List[BenchmarkData],
        directory_string: Optional[str] = None,
    ):
        self.benchmark_data = self._process_raw_benchmark_data(benchmark_data)
        self.directory_string = directory_string
        self.save_plots = False

    @classmethod
    def from_file(cls,
        file_name_parameters: Union[List[BoardGenerationParameters], List[str]],
        directory_string: Optional[str] = None,
    ):
        benchmark_data = []
        if isinstance(file_name_parameters[0], BoardGenerationParameters):
            for board_parameters in file_name_parameters:
                filename = \
                    f"{board_parameters.generator_type}_{board_parameters.rows}x" \
                    f"{board_parameters.columns}_agent_" \
                    f"{board_parameters.number_of_wires}.pkl"
                with open(directory_string + filename, "rb") as file:
                    benchmark = pickle.load(file)
                benchmark_data.append(benchmark)
        else:
            for file_name in file_name_parameters:
                with open(directory_string + file_name, "rb") as file:
                    benchmark = pickle.load(file)
                benchmark_data.append(benchmark)

        return cls(benchmark_data, directory_string)

    @classmethod
    def from_simulation(cls,
        benchmark_parameters_list: List[BoardGenerationParameters],
        number_of_runs: int,
        save_outputs: bool = False,
    ):
        # TODO (MW): Clean up this method
        benchmark_data = []
        benchmark_folder = None if not save_outputs else make_benchmark_folder(with_time=True)
        for board_parameters in benchmark_parameters_list:
            number_of_cells = board_parameters.rows * board_parameters.columns
            router = Route(
                instance_generator_type=board_parameters.generator_type,
                rows=board_parameters.rows,
                cols=board_parameters.columns,
                num_agents=board_parameters.number_of_wires,
                step_limit=number_of_cells,
            )
            simulation_outputs = router.route_for_benchmarking(number_of_boards=number_of_runs, **router.__dict__)
            benchmark = BenchmarkData(*simulation_outputs, generator_type=board_parameters)

            if save_outputs:
                filename = \
                    f"{board_parameters.generator_type}_" \
                    f"{board_parameters.rows}x{board_parameters.columns}_" \
                    f"agent_{board_parameters.number_of_wires}.pkl"

                with open(str(benchmark_folder)+f"/{filename}", "wb") as file:
                    pickle.dump(benchmark, file)
            benchmark_data.append(benchmark)

        return cls(benchmark_data, benchmark_folder)

    def plot_all(self, save_outputs: bool = True):
        self.save_plots = save_outputs
        if self.directory_string is None:
            self.directory_string = make_benchmark_folder()
        self.plot_rewards()
        self.plot_total_wire_lengths()
        self.plot_proportion_wires_connected()
        self.plot_number_of_steps()

    def plot_rewards(self):
        for board_size in self.benchmark_data.keys():
            mean_rewards = []
            labels = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_rewards.append(benchmark.average_reward_per_wire())

            file_name = f"/total_rewards_{board_size}" if self.save_plots else None
            self._plot_bar_chart(
                title=f"Total average rewards per wire over 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Total average rewards per wire",
                data=mean_rewards,
                labels=labels,
                file_name=file_name
            )

    def plot_total_wire_lengths(self):
        for board_size in self.benchmark_data.keys():
            mean_lengths = []
            labels = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_lengths.append(benchmark.average_total_wire_length())

            file_name = f"/total_wire_lenghts_{board_size}" if self.save_plots else None
            self._plot_bar_chart(
                title=f"Total average wire lengths over 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Total wire lengths",
                data=mean_lengths,
                labels=labels,
                file_name=file_name
            )

    def plot_proportion_wires_connected(self):
        for board_size in self.benchmark_data.keys():
            mean_proportion = []
            labels = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_proportion.append(benchmark.average_proportion_of_wires_connected())
            file_name = f"/proportion_of_wires_{board_size}" if self.save_plots else None
            self._plot_bar_chart(
                title=f"Proportion of wires connected over 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Proportion of wires connected",
                data=mean_proportion,
                labels=labels,
                file_name=file_name
            )

    def plot_number_of_steps(self):
        for board_size in self.benchmark_data.keys():
            mean_proportion = []
            labels = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_proportion.append(benchmark.average_steps_till_board_terminates())

            file_name = f"/steps_till_termination_{board_size}" if self.save_plots else None
            self._plot_bar_chart(
                title=f"Total average wire lengths over 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Average number of steps",
                data=mean_proportion,
                labels=labels,
                file_name=file_name,
            )

    def plot_number_of_filled_boards(self, save_fig: bool = True):
        for board_size in self.benchmark_data.keys():
            board_filled = []
            labels = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                number_of_boards_filled = jnp.sum(jnp.array(benchmark.was_board_filled))
                board_filled.append(number_of_boards_filled)

            file_name = f"/number_of_filled_boards{board_size}" if save_fig else None
            self._plot_bar_chart(
                title=f"Number of filled boards out of 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Number of filled boards",
                data=board_filled,
                labels=labels,
                file_name=file_name,
            )

    def _plot_bar_chart(self,
        x_label: str, y_label: str, title: str,
        data: Iterable, labels:  List[str],
        file_name: Optional[str] = None
    ):
        fig, ax = plt.subplots()
        ax.bar(labels, data)
        ax.set(
            title=title,
            xlabel=x_label,
            ylabel=y_label,
        )
        if file_name is not None:
            fig.savefig(str(self.directory_string) + str(file_name))
        else:
            plt.show()

    @staticmethod
    def _process_raw_benchmark_data(raw_benchmark_data):
        data_dict = {}

        for benchmark in raw_benchmark_data:
            board_params = str(benchmark.generator_type.rows) + "_" + str(
                benchmark.generator_type.columns) + "_" + str(
                benchmark.generator_type.number_of_wires)
            data_dict[board_params] = [benchmark] if data_dict.get(
                board_params) is None else data_dict.get(board_params) + [
                benchmark]

        for board_params in data_dict.keys():
            data_dict[board_params] = \
                sorted(data_dict[board_params],
                       key=lambda x:x.generator_type.generator_type)
        return data_dict



if __name__ == '__main__':

    benchmarks = [
        BoardGenerationParameters(rows=5, columns=5, number_of_wires=3,
                                  generator_type=BoardGenerators.BFS_2),


    ]

    # filenames = [
    #     "BFS 2_8x8_agent_5.pkl",
    #     "BFS_8x8_agent_5.pkl"
    # ]
    # dir_string = Path(__file__).parent.parent.parent
    # folder = "/ic_experiments/benchmarks/20230206_benchmark_14_03/"
    # directory = str(dir_string) + folder

    # benchmark = BasicBenchmark.from_file(filenames, directory)
    benchmark = BasicBenchmark.from_simulation(benchmarks, 1, False)
    print("bla")
    for board_size in benchmark.benchmark_data.keys():
        mean_proportion = []
        labels = []
        for benchmark in benchmark.benchmark_data[board_size]:
            labels.append(benchmark.generator_type.generator_type)



    # benchmark.plot_all(save_outputs=True)
