# from typing import Optional
#
# import numpy as np
#
# from ic_routing_board_generation.benchmarking.randy_route import board_generator
# from ic_routing_board_generation.ic_routing.route import Route
#
#
# class BenchmarkBoardGenerator:
#     def __init__(self,
#         empty_board: Optional[np.ndarray] = None,
#         filled_board: Optional[np.ndarray] = None,
#         fitted_wire: Optional[int] = None,
#     ):
#         self.empty_board = empty_board
#         self.filled_board = filled_board
#         self.number_of_fitted_wires = fitted_wire
#
#     @classmethod
#     def generate_randys_board(cls, x_dim: int, y_dim: int, target_wires: int):
#         empty_board, filled_board, number_of_wires = \
#             board_generator(x_dim, y_dim, target_wires)
#
#         return cls(empty_board, filled_board, number_of_wires)
#
#
# if __name__ == '__main__':
#     marta_board = BenchmarkBoardGenerator.generate_randys_board(6, 6, 3)
#     router = Route(instance_generator_type='marta',
#                          rows=25,
#                          cols=20,
#                          num_agents=12,
#                          step_limit=30,
#                          board=marta_board.empty_board
#                          )
#
#     router.route(time_steps=50, fps=15, **router.__dict__)
