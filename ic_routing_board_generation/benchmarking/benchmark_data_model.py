import dataclasses
from dataclasses import dataclass
from typing import List, Optional

from chex import Array
import jax.numpy as jnp

from ic_routing_board_generation.interface.board_generator_interface import \
    BoardName


@dataclass
class BoardGenerationParameters:
    rows: int
    columns: int
    number_of_wires: int
    generator_type: BoardName


@dataclass
class BenchmarkData:
    episode_length: List[float]
    episode_return: List[Array]
    num_connections: List[float]
    ratio_connections: List[float]
    time: List[float]
    total_path_length: List[int]
    generator_type: Optional[BoardGenerationParameters] = None

    def return_plotting_dict(self):
        plotting_dict = {
            "total_reward": {
                "x_label": "",
                "y_label": "",
                "bar_chart_title": "",
                "violin_plot_title": "",
                "average_value": "",
                "std": "",
                "file_name": "",

            }

        }
        return plotting_dict
    def average_reward_per_wire(self):
        return float(jnp.mean(jnp.array(self.episode_return), axis=0))

    def std_reward_per_wire(self):
        return float(jnp.std(jnp.array(self.episode_return), axis=(0)))

    def average_total_wire_length(self):
        return float(jnp.mean(jnp.array(self.total_path_length), axis=0))

    def std_total_wire_length(self):
        return float(jnp.std(jnp.array(self.total_path_length), axis=0))

    def average_proportion_of_wires_connected(self):
        return float(jnp.mean(jnp.array(self.ratio_connections), axis=0))

    def std_proportion_of_wires_connected(self):
        return float(jnp.std(jnp.array(self.ratio_connections), axis=0))

    def average_steps_till_board_terminates(self):
        return float(jnp.mean(jnp.array(self.episode_length), axis=0))

    def std_steps_till_board_terminates(self):
        return float(jnp.std(jnp.array(self.episode_length), axis=0))


if __name__ == '__main__':
    test = BenchmarkData
    # print(test.plotting_dict)
    for field in dataclasses.fields(BenchmarkData):
        print(field.name)
