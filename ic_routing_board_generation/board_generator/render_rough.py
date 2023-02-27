# Render boards from env_viewer.py
from jumanji.environments.combinatorial.routing.env_viewer import RoutingViewer
import numpy as np
from jax.numpy import asarray
from bfs_board import BFSBoard
from typing import Optional


def render_my_array(array: np.ndarray, num_agents: int, rows: int, columns: int, viewer_width: Optional[int] = 500,
                    viewer_height: Optional[int] = 500, save_img: Optional[str] = None):
    """ Render a board from a numpy array
    Args:
        array: The board to render
        num_agents: Number of agents in the environment.
        rows: Number of rows in the board.
        columns: Number of cols in the board.
        viewer_width: Width of the viewer in pixels.
        viewer_height: Height of the viewer in pixels.
        save_img: The name/folder of the image to save

    Returns:
        None
    """
    viewer = RoutingViewer(num_agents=num_agents, grid_rows=rows, grid_cols=columns, viewer_width=viewer_width,
                           viewer_height=viewer_height)
    if save_img is None:
        save_img = 'board_' + str(rows) + 'x' + str(columns) + 'x' + str(num_agents) + '.png'

    viewer.render(asarray(array), save_img=save_img)


if __name__ == '__main__':

    # ## EXAMPLES WITH UO BFSBOARDS
    # # Create an 8x8 board with 8 agents
    # board_rb = BFSBoard(num_agents=8, rows=8, columns=8, max_attempts=20)
    # # Fill Board with min_bends
    #
    # # Fill the board with 2 wires removed using the 'min_bends' method
    # board_rb.fill_board_with_clipping(2, 'min_bends', verbose=True)
    # print(board_rb.return_solved_board())
    # solved_board = board_rb.return_solved_board()
    # # print(board_rb.return_empty_board())
    #
    # viewer = RoutingViewer(num_agents=8, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    # viewer.render(asarray(solved_board), save_img='images/board_8x8x8.png')
    #
    # # Create an 8x8 board with 8 agents
    # board_rb = BFSBoard(num_agents=10, rows=8, columns=8, max_attempts=20)
    # # Fill Board with min_bends
    #
    # # Fill the board with 2 wires removed using the 'min_bends' method
    # board_rb.fill_board_with_clipping(2, 'min_bends', verbose=True)
    # print(board_rb.return_solved_board())
    # solved_board = board_rb.return_solved_board()
    # # print(board_rb.return_empty_board())
    #
    # viewer = RoutingViewer(num_agents=5, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    # viewer.render(asarray(solved_board), save_img='images/board_8x8x10.png')

    ###### Sample boards ####

    board_1 = np.array([[0, 2, 2, 2, 2, 2, 2, 2]
                           , [4, 2, 0, 2, 2, 0, 0, 2]
                           , [0, 0, 0, 16, 3, 2, 2, 2]
                           , [0, 12, 11, 14, 14, 0, 8, 10]
                           , [0, 0, 11, 11, 14, 14, 8, 8]
                           , [0, 0, 11, 11, 15, 14, 0, 8]
                           , [0, 0, 11, 13, 5, 5, 5, 8]
                           , [0, 0, 6, 5, 5, 7, 5, 9]])

    board_2 = np.array([[24, 23, 23, 23, 23, 23, 23, 23]
                           , [15, 14, 14, 14, 14, 14, 0, 23]
                           , [3, 2, 2, 2, 2, 14, 0, 23]
                           , [0, 20, 22, 0, 2, 14, 27, 25]
                           , [0, 21, 7, 6, 2, 14, 26, 12]
                           , [0, 10, 2, 2, 2, 14, 26, 11]
                           , [0, 8, 2, 0, 18, 16, 28, 11]
                           , [9, 8, 4, 0, 19, 31, 30, 13]])

    board_3 = np.array([[0, 7, 5, 0, 2, 2, 12, 13],
                        [0, 6, 5, 2, 2, 2, 2, 10],
                        [0, 5, 5, 2, 4, 2, 2, 8],
                        [0, 5, 5, 2, 2, 8, 8, 8],
                        [0, 5, 5, 0, 2, 8, 9, 15],
                        [0, 0, 2, 2, 2, 14, 16, 14],
                        [0, 3, 2, 0, 0, 14, 14, 14],
                        [0, 0, 0, 0, 0, 0, 14, 14]])

    viewer = RoutingViewer(num_agents=10, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    viewer.render(asarray(board_1), save_img='board_1.png')

    viewer = RoutingViewer(num_agents=10, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    viewer.render(asarray(board_2), save_img='board_2.png')

    viewer = RoutingViewer(num_agents=10, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    viewer.render(asarray(board_3), save_img='board_3.png')

    render_my_array(board_1, 10, 8, 8, 500, 500, 'board_1a.png')
    render_my_array(board_2, 10, 8, 8, 500, 500, 'board_2a.png')
    render_my_array(board_3, 10, 8, 8, 500, 500, 'board_3a.png')
