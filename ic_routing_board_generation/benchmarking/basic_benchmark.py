import pickle
from typing import List, Optional, Iterable, Union

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters, BenchmarkData
from ic_routing_board_generation.benchmarking.benchmark_utils import \
    make_benchmark_folder, files_list_from_benchmark_experiment
from ic_routing_board_generation.ic_routing.route import Route

# TODO (Marta): create master plotting loop for bar charts
# TODO (Marta): refactor [plotting functions into plotting utils

class BasicBenchmark:
    def __init__(
        self,
        benchmark_data: List[BenchmarkData],
        directory_string: str = None,
    ):
        self.benchmark_data = self._process_raw_benchmark_data(benchmark_data)
        self.directory_string = directory_string
        self.save_plots = False

    @classmethod
    def from_file(
        cls,
        file_name_parameters: Union[List[BoardGenerationParameters], List[str]],
        directory_string: Optional[str] = "",
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
            stds = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_rewards.append(benchmark.average_reward_per_wire())
                stds.append(benchmark.std_reward_per_wire())

            file_name = f"/total_rewards_{board_size}" if self.save_plots else None
            self._plot_bar_chart(
                title=f"Total average rewards per wire over 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Total average rewards per wire",
                data=mean_rewards,
                labels=labels,
                file_name=file_name,
                stds=stds
            )

    def plot_total_wire_lengths(self):
        for board_size in self.benchmark_data.keys():
            mean_lengths = []
            labels = []
            stds = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_lengths.append(benchmark.average_total_wire_length())
                stds.append(benchmark.std_total_wire_length())

            file_name = f"/total_wire_lenghts_{board_size}" if self.save_plots else None
            self._plot_bar_chart(
                title=f"Total average wire lengths over 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Total wire lengths",
                data=mean_lengths,
                labels=labels,
                file_name=file_name,
                stds=stds
            )

    def master_plotting_loop_violin(self):
        for board_size in self.benchmark_data.keys():
            for parameter in ["total_wire_lengths", "proportion_wires_connected", "number_of_steps"]:
                violin_chart_data = []
                labels = []
                for board_data in self.benchmark_data[board_size]:
                    violin_chart_data.append(board_data.__getattribute__(parameter))
                    labels.append(board_data.generator_type.generator_type.name)

                self._plot_violin_plot(
                    x_label="Board Generator", y_label=parameter,
                    title=f"{parameter} over 1000 boards, {board_size}",
                    data=np.array(violin_chart_data), labels=labels,
                    file_name=f"/dist_{parameter}_{board_size}",
                )

    def plot_proportion_wires_connected(self):
        for board_size in self.benchmark_data.keys():
            mean_proportion = []
            stds = []
            labels = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_proportion.append(benchmark.average_proportion_of_wires_connected())
                stds.append(benchmark.std_proportion_of_wires_connected())
            file_name = f"/proportion_of_wires_{board_size}" if self.save_plots else None
            self._plot_bar_chart(
                title=f"Proportion of wires connected over 1000 boards, {board_size}",
                x_label="Generator type",
                y_label="Proportion of wires connected",
                data=np.array(mean_proportion),
                labels=labels,
                file_name=file_name,
                stds=np.array(stds)
            )

    def plot_number_of_steps(self):
        for board_size in self.benchmark_data.keys():
            mean_proportion = []
            labels = []
            stds = []
            for benchmark in self.benchmark_data[board_size]:
                labels.append(benchmark.generator_type.generator_type.value)
                mean_proportion.append(benchmark.average_steps_till_board_terminates())
                stds.append(benchmark.std_steps_till_board_terminates())

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

    def _plot_bar_chart(
        self,
        x_label: str, y_label: str, title: str,
        data: Iterable, labels:  List[str],
        file_name: Optional[str] = None,
        stds = None,
    ):
        fig, ax = plt.subplots()
        ax.bar(labels, data)
        ax.set(
            title=title,
            xlabel=x_label,
            ylabel=y_label,
        )
        data = np.array(data)

        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        inds = np.arange(0, len(data))
        ax.scatter(inds, data, marker='o', color='k', s=30, zorder=3)
        if stds is not None:
            stds = np.array(stds)
            ax.vlines(inds, data - (stds/2), data + (stds/2), color='blue', linestyle='-',
                  lw=3)
        plt.tight_layout()
        if file_name is not None:
            fig.savefig(str(self.directory_string) + str(file_name))
        else:
            plt.show()

    def _plot_violin_plot(self,
        x_label: str, y_label: str, title: str,
        data: Iterable, labels: List[str],
        file_name: Optional[str] = None
    ):
        fig, ax = plt.subplots()
        ax.violinplot(data, showmeans=True)
        ax.set(
            title=title,
            ylabel=y_label,
        )

        # quartile1, medians, quartile3 = np.percentile(data,
        #                                               [25, 50, 75], axis=1)
        # whiskers = np.array([
        #     self._adjacent_values(sorted_array, q1, q3)
        #     for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        #
        # inds = np.arange(1, len(medians) + 1)
        # ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-',
        #            lw=1)
        # means = np.mean(data, axis =1)
        # inds = np.arange(1, len(medians) + 1)
        # # ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        # ax.scatter(inds, means, marker='x', color='k', s=30, zorder=3)
        # ax.vlines(inds, quartile1, quartile3, color='blue', linestyle='-', lw=3)

        self._set_axis_style(ax, labels, x_label)
        plt.tight_layout()

        if file_name is not None and self.save_plots:
            fig.savefig(str(self.directory_string) + str(file_name))
        else:
            plt.show()


    def _adjacent_values(self, vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value

    def _set_axis_style(self, ax, labels: List[str], x_axis_label: str = "Board gen"):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation=30, ha='right')
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel(x_axis_label)

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
    directory = "20230208_benchmark_23_45"
    all_files = files_list_from_benchmark_experiment(directory)
    benchmark = BasicBenchmark.from_file(all_files, directory_string=directory + "/")
    print()


    # benchmarks = [
    #     BoardGenerationParameters(rows=5, columns=5, number_of_wires=3,
    #                               generator_type=BoardName.BFS_2),
    #
    #
    # ]
    #
    # # filenames = [
    # #     "BFS 2_8x8_agent_5.pkl",
    # #     "BFS_8x8_agent_5.pkl"
    # # ]
    # # dir_string = Path(__file__).parent.parent.parent
    # # folder = "/ic_experiments/benchmarks/20230206_benchmark_14_03/"
    # # directory = str(dir_string) + folder
    #
    # # benchmark = BasicBenchmark.from_file(filenames, directory)
    # benchmark = BasicBenchmark.from_simulation(benchmarks, 1, False)
    # print("bla")
    # for board_size in benchmark.benchmark_data.keys():
    #     mean_proportion = []
    #     labels = []
    #     for benchmark in benchmark.benchmark_data[board_size]:
    #         labels.append(benchmark.generator_type.generator_type)



    # benchmark.plot_all(save_outputs=True)
