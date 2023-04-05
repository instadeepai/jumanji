# Render boards from env_viewer.py
from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard
from ic_routing_board_generation.visualisation.env_viewer import RoutingViewer
import numpy as np
from jax.numpy import asarray
from bfs_board import BFSBoard
from typing import Optional, Union
import os


def render_my_array(array: Union[AbstractBoard, np.ndarray], num_agents: int, rows: int, columns: int,
                    viewer_width: Optional[int] = 500,
                    viewer_height: Optional[int] = 500, save_img: Optional[str] = None):
    """ Render a board from a board object or a numpy array. If a board object is passed, both the solved board and
    training board will be rendered.
    Args:
        array: The board to render
        num_agents: Number of agents in the environment.
        rows: Number of rows in the board.
        columns: Number of cols in the board.
        viewer_width: Width of the viewer in pixels.
        viewer_height: Height of the viewer in pixels.
        save_img: The name/folder of the image to save. If None, it will be saved as
                board_{rows}x{columns}x{num_agents}.png. It does not overwrite but appends a number to the end.

    Returns:
        None
    """
    if isinstance(array, AbstractBoard):
        solved_array = array.return_solved_board()
        training_array = array.return_training_board()
        renders = [solved_array, training_array]
        prefixes = ['solved', 'training']
    else:
        renders = [array]
        prefixes = ['']

    for i, render in enumerate(renders):
        prefix = prefixes[i]
        viewer = RoutingViewer(num_agents=num_agents, grid_rows=rows, grid_cols=columns, viewer_width=viewer_width,
                               viewer_height=viewer_height)
        save_img_ = save_img

        if save_img_ is None:
            save_img_ = 'board_' + str(rows) + 'x' + str(columns) + 'x' + str(num_agents) + '.png'
        else:
            # Remove the folder name and add it to the prefix
            if '/' in save_img_:  # If there is a folder in the name, only take the file name and the folders to the prefix
                prefix = save_img_[:save_img_.rfind('/')] + '/' + prefix
                save_img_ = save_img_[save_img_.rfind('/') + 1:]

        if prefix != '' and len(prefix) > 1:
            save_img_ = prefix + '_' + save_img_
        else:
            save_img_ = prefix + save_img_
        # If save exists in location, append a number to the end or increments the number

        if os.path.exists(save_img_):
            num = 1
            while os.path.exists(save_img_):
                # Look for _{num}.png in the name
                if '_' + str(num) + '.png' in save_img_:  # If the number is already in the name, increment it
                    num += 1
                    save_img_ = save_img_.replace('_' + str(num - 1) + '.png', '_' + str(num) + '.png')
                else:  # If the number is not in the name, add it
                    save_img_ = save_img_.replace('.png', '_' + str(num) + '.png')

        viewer.render(asarray(render), save_img=save_img_)


if __name__ == '__main__':
    ### Example Usage

    ## 1. Render a numpy array

    # Create the numpy array
    board_1 = np.array([[0, 2, 2, 2, 2, 2, 2, 2],
                        [4, 2, 0, 2, 2, 0, 0, 2],
                        [0, 0, 0, 16, 3, 2, 2, 2],
                        [0, 12, 11, 14, 14, 0, 8, 10],
                        [0, 0, 11, 11, 14, 14, 8, 8],
                        [0, 0, 11, 11, 15, 14, 0, 8],
                        [0, 0, 11, 13, 5, 5, 5, 8],
                        [0, 0, 6, 5, 5, 7, 5, 9]])

    # Save it as board_1.png
    render_my_array(board_1, 10, 8, 8, 500, 500, 'board_1.png')

    # Save it as board_8x8x10.png
    render_my_array(board_1, 10, 8, 8, 500, 500)

    # 2. Render a board object
    # Create a board object
    board_2 = BFSBoard(num_agents=10, rows=8, columns=8, max_attempts=20)
    board_2.fill_board_with_clipping(2, 'min_bends', verbose=False)

    # Render the solved board and save it as board_2.png
    render_my_array(board_2, 10, 8, 8, 500, 500, 'board_2.png')

    # board_2 = np.array([[24, 23, 23, 23, 23, 23, 23, 23],
    #                     [15, 14, 14, 14, 14, 14, 0, 23],
    #                     [3, 2, 2, 2, 2, 14, 0, 23],
    #                     [0, 20, 22, 0, 2, 14, 27, 25],
    #                     [0, 21, 7, 6, 2, 14, 26, 12],
    #                     [0, 10, 2, 2, 2, 14, 26, 11],
    #                     [0, 8, 2, 0, 18, 16, 28, 11],
    #                     [9, 8, 4, 0, 19, 31, 30, 13]])
    #
    # board_3 = np.array([[0, 7, 5, 0, 2, 2, 12, 13],
    #                     [0, 6, 5, 2, 2, 2, 2, 10],
    #                     [0, 5, 5, 2, 4, 2, 2, 8],
    #                     [0, 5, 5, 2, 2, 8, 8, 8],
    #                     [0, 5, 5, 0, 2, 8, 9, 15],
    #                     [0, 0, 2, 2, 2, 14, 16, 14],
    #                     [0, 3, 2, 0, 0, 14, 14, 14],
    #                     [0, 0, 0, 0, 0, 0, 14, 14]])
    #
    # viewer = RoutingViewer(num_agents=10, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    # viewer.render(asarray(board_1), save_img='board_1.png')
    #
    # viewer = RoutingViewer(num_agents=10, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    # viewer.render(asarray(board_2), save_img='board_2.png')
    #
    # viewer = RoutingViewer(num_agents=10, grid_rows=8, grid_cols=8, viewer_width=500, viewer_height=500)
    # viewer.render(asarray(board_3), save_img='board_3.png')
    #
    # # render_my_array(board_1, 10, 8, 8, 500, 500, 'board_1a.png')
    # # render_my_array(board_2, 10, 8, 8, 500, 500, 'board_2a.png')
    # # render_my_array(board_3, 10, 8, 8, 500, 500, 'board_3a.png')