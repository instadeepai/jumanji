import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np


class Grid:
    def __init__(self, rows: int, cols: int, grid_layout: Optional[np.ndarray] = None) -> None:
        """
        Constructor for the Grid class.
        Args:
            rows: number of rows in the grid
            cols: number of columns in the grid
            grid_layout: grid layout
        Returns:
            None
        """
        self.rows = rows
        self.cols = cols
        if grid_layout is None:
            self.layout = np.zeros((rows, cols), dtype=np.int32)
        else:
            self.layout = grid_layout
        self.path = []
        self.visited = np.full((rows, cols), False)

    def bfs_maze(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Function to find the shortest path between two points in a grid using BFS.
        Args:
            start: start position
            end: end position
        Returns:
            tuple containing whether a path exists, the path,
            and the number of steps
        """
        # Initialize queue, visited set, and path dictionary
        queue = deque([start])
        visited = {start: None}
        steps = {start: 0}

        # Define possible movements
        row = [-1, 0, 1, 0]
        col = [0, 1, 0, -1]

        # Loop through the queue
        while queue:
            # Get the current position
            curr_pos = queue.popleft()

            # Check if we have reached the end
            if curr_pos == end:
                # retrace the path from end to start
                curr = end
                path = [curr]
                while curr != start:
                    curr = visited[curr]
                    path.append(curr)
                return True, path[::-1], steps[end]

            # Randomly Loop through possible movements
            inds = list(range(4))
            random.shuffle(list(range(4)))
            for i in inds:
                # Calculate new position
                new_row = curr_pos[0] + row[i]
                new_col = curr_pos[1] + col[i]

                # Check if the new position is valid and not visited only valid positions have value 0
                if (0 <= new_row < self.layout.shape[0]) and (0 <= new_col < self.layout.shape[1]) and (
                        self.layout[new_row, new_col] == 0) and (new_row, new_col) not in visited:
                    # Add the new position to the queue and mark it as visited
                    queue.append((new_row, new_col))
                    visited[(new_row, new_col)] = curr_pos
                    steps[(new_row, new_col)] = steps[curr_pos] + 1
        return False, [], 0

        #         # Check if the new position is valid and not visited
        #         if (0 <= new_row < self.layout.shape[0]) and (0 <= new_col < self.layout.shape[1]) and (
        #                 self.layout[new_row, new_col] != 1) and (new_row, new_col) not in visited:
        #             # Add the new position to the queue and mark it as visited
        #             queue.append((new_row, new_col))
        #             visited[(new_row, new_col)] = curr_pos
        #             steps[(new_row, new_col)] = steps[curr_pos] + 1
        # return False, [], 0

    def fill_grid(self, path: List[Tuple[int, int]], str_num: int = None, fill_num: int = None,
                  end_num: int = None) -> None:
        """
        Function to fill the grid with the path.
        Args:
            path: path to fill
            str_num: number to fill the start position with
            fill_num: number to fill the path with
            end_num: number to fill the end position with
        Returns:
            None
        """
        if str_num is None:
            str_num = end_num = fill_num = 1
        for i, pt in enumerate(path):
            if self.layout[pt[0]][pt[1]] != 0 and self.layout[pt[0]][pt[1]] != 1:
                print("Replacing {} with {}".format(self.layout[pt[0]][pt[1]], fill_num))
                raise ValueError("ERROR: Path already exists at position: {}".format(pt))

            if i == 0:
                self.layout[pt[0]][pt[1]] = str_num
            elif i == len(path) - 1:
                self.layout[pt[0]][pt[1]] = end_num
            else:
                self.layout[pt[0]][pt[1]] = fill_num

    def remove_path(self, path: List[Tuple[int, int]]) -> None:
        """
        Function to remove a path from the grid.
        Args:
            path: path to remove
        Returns:
            None
        """
        for i, pt in enumerate(path):
            self.layout[pt[0]][pt[1]] = 0

    def reset_maze(self) -> None:
        """
        Function to reset the maze. Called after a partial fill and resets previously filled paths to 1.
        Args:
            None
        Returns:
            None
        """
        self.layout[self.layout != 0] = 1
