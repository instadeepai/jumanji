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
    total_reward: List[Array]
    was_board_filled: List[bool]
    total_wire_lengths: List[int]
    proportion_wires_connected: List[float]
    number_of_steps: List[int]
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
        return jnp.mean(jnp.array(self.total_reward), axis=(0, 1))

    def std_reward_per_wire(self):
        return jnp.std(jnp.array(self.total_reward), axis=(0, 1))

    def average_total_wire_length(self):
        return jnp.mean(jnp.array(self.total_wire_lengths), axis=0)

    def std_total_wire_length(self):
        return jnp.std(jnp.array(self.total_wire_lengths), axis=0)

    def average_proportion_of_wires_connected(self):
        return jnp.mean(jnp.array(self.proportion_wires_connected), axis=0)

    def std_proportion_of_wires_connected(self):
        return jnp.std(jnp.array(self.proportion_wires_connected), axis=0)

    def average_steps_till_board_terminates(self):
        return jnp.mean(jnp.array(self.number_of_steps), axis=0)

    def std_steps_till_board_terminates(self):
        return jnp.std(jnp.array(self.number_of_steps), axis=0)



# TODO combine into 1 dataclass?

if __name__ == '__main__':
    test = BenchmarkData
    # print(test.plotting_dict)
    for field in dataclasses.fields(BenchmarkData):
        print(field.name)
