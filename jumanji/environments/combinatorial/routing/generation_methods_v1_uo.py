import random
from collections import deque
from typing import List, Tuple, Union, Optional, Dict, Callable
import numpy as np


class Grid:
    def __init__(self, rows: int, cols: int, grid: Optional[np.ndarray] = None) -> None:
        """
        Constructor for the Grid class.
        Args:
            rows (int): number of rows in the grid
            cols (int): number of columns in the grid
            grid (np.array): grid layout
        Returns:
            None
        """
        self.rows = rows
        self.cols = cols
        if grid is None:
            self.layout = np.zeros((rows, cols), dtype=np.int32)
        else:
            self.layout = grid
        self.path = []
        self.visited = np.full((rows, cols), False)

    def bfs_maze(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Function to find the shortest path between two points in a grid using BFS.
        Args:
            start (tuple[int, int]): start position
            end (tuple[int, int]): end position
        Returns:
            tuple[bool, list[tuple[int, int]], int]: tuple containing whether a path exists, the path,
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

                # Check if the new position is valid and not visited
                if (0 <= new_row < self.layout.shape[0]) and (0 <= new_col < self.layout.shape[1]) and (
                        self.layout[new_row, new_col] != 1) and (new_row, new_col) not in visited:
                    # Add the new position to the queue and mark it as visited
                    queue.append((new_row, new_col))
                    visited[(new_row, new_col)] = curr_pos
                    steps[(new_row, new_col)] = steps[curr_pos] + 1
        return False, [], 0

    def fill_grid(self, path: List[Tuple[int, int]], str_num: int = None, fill_num: int = None,
                  end_num: int = None) -> None:
        """
        Function to fill the grid with the path.
        Args:
            path (list[tuple[int, int]]): path to fill
            str_num (int): number to fill the start position with
            fill_num (int): number to fill the path with
            end_num (int): number to fill the end position with
        Returns:
            None
        """
        if str_num is None:
            str_num = end_num = fill_num = 1
        for i, pt in enumerate(path):
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
            path (list[tuple[int, int]]): path to remove
        Returns:
            None
        """
        for i, pt in enumerate(path):
            self.layout[pt[0]][pt[1]] = 0


class BFSBoard:
    def __init__(self, rows: int, columns: int, num_agents: int, max_attempts: int = 10) -> None:
        """Constructor for the Board class.

        Args:
            rows (int): number of rows in the board
            columns (int): number of columns in the board
            num_agents (int): number of wires in the board
            max_attempts (int): maximum number of attempts to fill the board
        Returns:
            None
            """
        self.rows = rows
        self.columns = columns
        self.grid = Grid(rows, columns)
        self.wires = num_agents
        self.paths = []
        self.starts = []
        self.ends = []
        self.filled = False
        self.solved_board = None
        self.empty_board = None
        self.partial_board = None  # board with  (<num_agents) wires filled (only populated if fill is unsuccessful)
        self.unsolved_board = None  # partial board with wires removed (only populated if fill is unsuccessful)
        self.max_attempts = max_attempts
        self.filled_wires = 0
        self.clip_method_dict = self.get_clip_method_dict()

    def pick_start_end(self, min_dist: Optional[int] = None) -> Tuple:
        """Picks a random start and end point for a wire.
        Args:
            min_dist (int): minimum distance between start and end points
        Returns:
            tuple: start and end points
        """
        options = np.argwhere(self.grid.layout == 0).tolist()
        assert len(options) >= 2, "Not enough empty spaces to place a wire."
        points = random.sample(options, 2)
        if min_dist is not None:
            while (abs(points[0][0] - points[1][0]) + abs(points[0][1] - points[1][1])) ** 0.5 < min_dist:
                points = random.sample(options, 2)
        return tuple(points[0]), tuple(points[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[bool, List[Tuple[int, int]], int]:
        """Places wire on the board if possible and returns a BFS path.
        Args:
            start (tuple): start point
            end (tuple): end point
        Returns:
            tuple: whether a path was found, the path, and the number of steps
            """
        found, path, num = self.grid.bfs_maze(start, end)
        return found, path, num

    def fill_board(self, verbose=False) -> Tuple[np.ndarray, np.ndarray, int]:
        """Fills the board with wires.
        Returns:
            tuple: the filled board, the empty board, and the number of wires
        """
        self.place_wires(verbose=verbose)

        return self.return_boards(verbose=verbose)

    def populate_grid(self, hard_fill: bool = False) -> None:
        """Populates the grid with the correct numbers.
        Heads: encoded as 4, 7 , 10,... (x%3==1)
        Targets: encoded as 3, 6, 9,... (y%3==0)
        Routes: encoded as 2, 5, 8,... (z%3==2)

        Args:
            hard_fill (bool): whether to hard fill the grid
        Returns:
            None
        """
        if not hard_fill:
            assert self.filled, "Cannot populate the grid if the board is not filled!"
            assert len(self.paths) == self.wires, "Something's wrong. Number of paths don't match the wires"
        self.shuffle_all()
        for k, path in enumerate(self.paths):
            head = 3 * (k + 1) + 1
            target = 3 * (k + 1)
            route = 3 * (k + 1) - 1
            self.grid.fill_grid(path, str_num=head, fill_num=route, end_num=target)

    def remove_routes(self, input_board: Optional[np.ndarray] = None) -> np.ndarray:
        """Removes the routes from the board.
        Args:
            input_board (np.array): the board to remove the routes from
        Returns:
            np.array: the board with the routes removed
        """
        if input_board is None:
            self.empty_board = np.where(self.solved_board % 3 != 2, self.solved_board, 0)
            return self.empty_board
        else:
            return np.where(input_board % 3 != 2, input_board, 0)

    def append_all(self, start: Tuple[int, int], end: Tuple[int, int], path: List[Tuple[int, int]]) -> None:
        """Appends the start, end and path to the corresponding lists.
        Args:
            start (tuple): start point
            end (tuple): end point
            path (list): path
        Returns:
            None
        """
        self.starts.append(start)
        self.ends.append(end)
        self.paths.append(path)

    def reset_board(self) -> None:
        """Resets the board.
        Returns:
            None
        """
        # reset grid, starts, ends and paths
        self.grid = Grid(self.rows, self.columns)
        self.paths = []
        self.starts = []
        self.ends = []
        self.filled = False
        self.solved_board = None
        self.empty_board = None
        self.partial_board = None
        self.unsolved_board = None
        self.filled_wires = 0

    def shuffle_all(self) -> None:
        """Shuffles the starts, ends and paths.
        Returns:
            None
        """
        # shuffle all
        zipped = list(zip(self.starts, self.ends, self.paths))
        random.shuffle(zipped)
        self.starts, self.ends, self.paths = zip(*zipped)

    def remove_wire(self, wire: int) -> None:
        """Removes a wire from the board.
        Args:
            wire (int): index of the wire to remove
        Returns:
            None
        """
        # remove wire from all lists
        self.starts.pop(wire)
        self.ends.pop(wire)
        self.paths.pop(wire)
        self.filled_wires -= 1

    @staticmethod
    def count_bends(path: List[tuple]) -> int:
        """Counts the number of bends in a path.
        Args:
            path (list[tuple]): the path to count the bends in
        Returns:
            int: the number of bends in the path
        """
        bends = 0
        for i in range(1, len(path) - 1):
            if path[i - 1][0] != path[i + 1][0] and path[i - 1][1] != path[i + 1][1]:
                bends += 1
        return bends

    def sort_by_bends(self) -> None:
        """Sorts the paths by the number of bends.
        Returns:
            None
        """
        bends = [self.count_bends(path) for path in self.paths]
        zipped = list(zip(bends, self.starts, self.ends, self.paths))

        # Sort by bends and then length if ties
        zipped.sort(key=lambda x: (x[0], len(x[3])))
        # zipped.sort(key=lambda x: x[0])
        bends, self.starts, self.ends, self.paths = map(list, (zip(*zipped)))

    def sort_by_length(self) -> None:
        """Sorts the paths by the length.
        Returns:
            None
        """
        lengths = [len(path) for path in self.paths]
        zipped = list(zip(lengths, self.starts, self.ends, self.paths))
        zipped.sort(key=lambda x: x[0])
        lengths, self.starts, self.ends, self.paths = map(list, (zip(*zipped)))

    def remove_fewest_bends(self, num: int) -> None:
        """Removes the wires with the fewest bends.
        Args:
            num (int): number of wires to remove
        Returns:
            None
        """
        self.sort_by_bends()
        extra_wires = min(num, len(self.paths))
        for _ in range(extra_wires):
            self.grid.remove_path(self.paths[0])
            self.remove_wire(0)

    def remove_shortest(self, num: int) -> None:
        """Removes the shortest wires.
        Args:
            num (int): number of wires to remove
        Returns:
            None
        """
        self.sort_by_length()
        extra_wires = min(num, len(self.paths))
        for _ in range(extra_wires):
            self.grid.remove_path(self.paths[0])
            self.remove_wire(0)

    def remove_longest(self, num: int) -> None:
        """Removes the longest wires.
        Args:
            num (int): number of wires to remove
        Returns:
            None
        """
        self.sort_by_length()
        extra_wires = min(num, len(self.paths))
        for _ in range(extra_wires):
            self.grid.remove_path(self.paths[-1])
            self.remove_wire(-1)

    def remove_random(self, num: int) -> None:
        """Removes random wires.
            Args:
                num (int): number of wires to remove
            Returns:
                None
            """
        extra_wires = min(num, len(self.paths))
        for _ in range(extra_wires):
            wire = random.randint(0, len(self.paths) - 1)
            self.grid.remove_path(self.paths[wire])
            self.remove_wire(wire)

    def remove_k_bends(self, k: int) -> None:
        """Removes all wires with fewer than k bends.
        Args:
            k (int): number of bends
        Returns:
            None
        """
        self.sort_by_bends()
        while self.count_bends(self.paths[0]) < k and len(self.paths) > 1:
            self.grid.remove_path(self.paths[0])
            self.remove_wire(0)

    def remove_k_length(self, k: Optional[int] = 3) -> None:
        """ Removes all wires of length less than k"""
        self.sort_by_length()
        while len(self.paths[0]) < k and len(self.paths) > 1:
            self.grid.remove_path(self.paths[0])
            self.remove_wire(0)

    def remove_first_k(self, k: int) -> None:
        """Removes the first k wires.
        Args:
            k (int): number of wires to remove
        Returns:
            None
        """
        # Check that there are enough wires to remove
        extra_wires = min(k, len(self.paths))
        for _ in range(k):
            self.grid.remove_path(self.paths[0])
            self.remove_wire(0)

    def get_clip_method_dict(self) -> Dict[str, Callable]:
        """Creates a dictionary of the methods.
        Returns:
            dict[str, function]: the dictionary of methods
        """
        return dict(min_bends=self.remove_fewest_bends, shortest=self.remove_shortest, longest=self.remove_longest,
                    random=self.remove_random, bends=self.remove_k_bends, fifo=self.remove_first_k)

    def clip(self, method: str, num: int) -> None:
        """Removes wires using the given method.
        Args:
            method (str): the method to use
            num (int): the number of wires to remove
        Returns:
            None
        """
        try:
            clip_func = self.clip_method_dict[method]
        except KeyError:
            raise ValueError(f"Clip Method {method} not found.")

        # Remove the wires using the given method
        clip_func(num)

    def pick_and_place(self, i: int = 0, verbose: Optional[bool] = False) -> tuple[bool, int]:
        """Picks a random start and end point and tries to place a wire.
        Args:
            i (int): index of the wire to place
            verbose (bool): whether to print the progress
        Returns:
            tuple[bool, list[tuple], int]: if the wire was successfully placed, the path and the number of filled wires
        """
        attempts = 0
        found = False
        while not found and attempts < self.max_attempts:
            if verbose:
                print(f"Fitting wire {i + 1} - attempt {attempts + 1}")
            attempts += 1
            # first pick random start and end
            start, end = self.pick_start_end()
            # then try to fill board
            found, path, _ = self.find_path(start, end)

            if found:
                # a path has been successfully found
                self.append_all(start, end, path)
                self.grid.fill_grid(path)
                self.filled_wires += 1

        return found, self.filled_wires

    def place_wires(self, verbose: Optional[bool] = False) -> None:
        """Places wires on the board.
        Args:
            verbose (bool): whether to print the progress
        Returns:
            None
        """
        i = self.filled_wires
        found = True

        while i < self.wires and found:
            found, _ = self.pick_and_place(i, verbose=verbose)
            i += 1

        return None

    def announce_failure(self):
        """Announces that board unable to be filled after max attempts."""
        print(
            f'Fill unsuccessful for wire {self.filled_wires + 1} after {self.max_attempts} attempts. '
            f'\n Returning {self.rows}x{self.columns} board with {self.filled_wires} wire(s).')

    def announce_success(self):
        """Announces that board has been filled."""
        print(f'Fill successful. Returning {self.rows}x{self.columns} board with {self.filled_wires} wire(s)')

    def return_boards(self, verbose: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray, int]:
        """Returns the board and the solved board.
        Returns:
            tuple[np.array, np.array]: the board, the solved board
        """
        assert self.filled_wires <= self.wires, "Too many wires have been placed."
        if self.filled_wires < self.wires:
            if verbose:
                self.announce_failure()
            self.populate_grid(hard_fill=True)
            self.partial_board = self.grid.layout.copy()
            self.unsolved_board = self.remove_routes(self.partial_board)
            return self.partial_board, self.unsolved_board, self.filled_wires
        else:
            if verbose:
                self.announce_success()
            self.filled = True
            self.populate_grid()
            self.solved_board = self.grid.layout.copy()
            self.remove_routes()
            return self.solved_board, self.empty_board, self.wires

    def fill_board_with_clipping(self, num_clips: int = 2, method: str = 'bends', verbose: Optional[bool] = False) -> \
            Tuple[np.ndarray, np.ndarray, int]:
        """Fills the board, clips the wires and then tries to fill again.
        Args:
            num_clips (int): number of wires to remove
            method (str): method to remove the wires
            verbose (bool): if True, prints the progress
        Returns:
            tuple[np.array, np.array, int]: the board, the solved board and the number of filled wires
        """
        self.place_wires(verbose=verbose)
        self.clip(method, num_clips)

        # Try to fill again
        self.place_wires(verbose=verbose)

        return self.return_boards(verbose=verbose)

    def return_empty_board(self) -> np.ndarray:
        """Returns the empty board with heads and targets but not wires.
        Returns:
            np.array: the empty board
        """
        if self.filled:
            return self.empty_board
        else:
            return self.unsolved_board

    def return_solved_board(self) -> np.ndarray:
        """Returns the solved board with heads, targets and wires.
        Returns:
            np.array: the solved board
        """
        if self.filled:
            return self.solved_board
        else:
            return self.partial_board

    def fill_clip_fill(self, num_clips: Union[int, List[int]], methods: Union[str, List[str]],
                       num_loops: Optional[int] = 1, verbose: Optional[bool] = False) -> \
            Tuple[np.ndarray, np.ndarray, int]:
        """ Performs a number of fill, clip, fill loops.
        Args:
            num_clips (int): number of wires to remove
            methods (str): method to remove the wires
            num_loops (int): number of loops to perform
            verbose (bool): if True, prints the progress
        Returns:
            tuple[np.array, np.array, int]: the board, the solved board and the number of filled wires
        """
        if isinstance(num_clips, int):
            num_clips = [num_clips] * num_loops
        if isinstance(methods, str):
            methods = [methods] * num_loops
        assert len(num_clips) == len(methods), "Number of clips and methods must be the same."
        if verbose:
            print(f"Performing {num_loops} loops of fill-clip-fill.")
        for i in range(num_loops):
            self.place_wires(verbose=verbose)
            self.clip(methods[i], num_clips[i])

        # Perform the final fill
        self.place_wires(verbose=verbose)

        # Return the boards
        return self.return_boards(verbose=verbose)


if __name__ == '__main__':
    # Example usage
    # Generate a board with 10 rows, 10 columns, 10 wires (num_agents) and with max 10 attempts to place each wire
    board = BFSBoard(rows=10, columns=10, num_agents=10, max_attempts=10)

    # Perform a standard fill
    board.fill_board(verbose=True)
    print(board.return_solved_board())
    print(board.return_empty_board())

    # Reset the board
    board.reset_board()
    # Fill the board with 2 wires removed using the 'min_bends' method
    board.fill_board_with_clipping(2, 'min_bends', verbose=True)
    print(board.return_solved_board())
    print(board.return_empty_board())

    # Reset the board
    board.reset_board()
    # Fill the board with 2 wires removed using the 'min_bends' method and 2 wires removed using the 'random' method
    board.fill_clip_fill([2, 2], ['min_bends', 'random'], num_loops=2, verbose=True)
    print(board.return_solved_board())
    print(board.return_empty_board())

    # Render the board
    
