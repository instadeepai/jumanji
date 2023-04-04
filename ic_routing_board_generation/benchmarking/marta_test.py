from typing import Tuple
import jax.numpy as jnp
import numpy as np


def get_heads_and_targets(board_layout: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the heads and targets of the board layout
    heads are encoded as 2,5,8,11,...
    targets are encoded as 3,6,9,12,...
    both are greater than 0
    """
    # Get the heads and targets in the right order
    heads = []
    targets = []
    # Get the maximum value in the board layout
    max_val = np.max(board_layout)
    # Get the heads and targets
    for i in range(1, max_val + 1):
        # Get the head and target
        if i % 3 == 2:
            head = np.argwhere(board_layout == i)[0]
            print(head)
            heads.append(tuple(head))
            target = np.argwhere(board_layout == i + 1)[0]
            print(target)
            targets.append(tuple(target))

    # convert heads and targets to horizontally stacked jax arrays and transpose
    heads = jnp.array(heads).T
    targets = jnp.array(targets).T

    return heads, targets


if __name__ == '__main__':
    # Sample usage
    # Create the numpy array
    board = np.array([[22, 20, 20, 15, 2, 2, 3, 0, 0, 0],
                      [13, 0, 20, 14, 2, 30, 29, 29, 29, 29],
                      [11, 0, 20, 14, 2, 2, 4, 28, 27, 29]
                         , [11, 0, 20, 14, 14, 14, 14, 9, 8, 29]
                         , [11, 0, 20, 20, 21, 0, 16, 0, 8, 29]
                         , [11, 7, 5, 5, 5, 5, 5, 5, 8, 29]
                         , [11, 11, 11, 11, 11, 12, 0, 6, 8, 29]
                         , [0, 18, 19, 0, 0, 8, 8, 8, 8, 29]
                         , [0, 0, 0, 0, 0, 10, 25, 0, 0, 29]
                         , [0, 0, 0, 0, 24, 23, 23, 0, 0, 31]])

    # take one away from each non-zero value in the array so it fits the new encoding
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] != 0:
                board[i, j] -= 1

    lists = get_heads_and_targets(board)
    # Print each item in lists
    for item in lists:
        print(item)
