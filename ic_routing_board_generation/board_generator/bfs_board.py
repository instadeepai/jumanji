import random
from typing import List, Tuple, Union, Optional, Dict, Callable
import numpy as np
from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard
from ic_routing_board_generation.board_generator.grid import Grid


class BFSBoard(AbstractBoard):
    def __init__(self, rows: int, columns: int, num_agents: int, max_attempts: int = 10) -> None:
        """Constructor for the Board class.

        Args:
            rows: number of rows in the board
            columns: number of columns in the board
            num_agents: maximum number of wires to place on the board
            max_attempts: maximum number of attempts to fill the board
        Returns:
            None
            """
        super().__init__(rows=rows, cols=columns, num_agents=num_agents)
        self.rows = rows
        self.columns = columns
        self.grid = Grid(rows, columns)
        self.num_agents = num_agents
        self.paths = []
        self.starts = []
        self.ends = []
        self.filled = False
        self.solved_board = None
        self.empty_board = None
        self.partial_board = None  # board with  (<num_agents) wires filled (only populated if fill is unsuccessful)
        self.unsolved_board = None  # partial board with wires removed (only populated if fill is unsuccessful)
        self.max_attempts = max_attempts
        self._wires_on_board = 0
        self.clip_method_dict = self.get_clip_method_dict()
        # self.fill_board(verbose=False)

    def pick_start_end(self, min_dist: Optional[int] = None) -> Tuple:
        """Picks a random start and end point for a wire.
        Args:
            min_dist: minimum distance between start and end points
        Returns:
            tuple containing start and end points
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
            start: start point
            end: end point
        Returns:
            tuple containing whether a path was found, the path, and the number of steps
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
        Routes: encoded as 1, 4, 7,... (x%3==1)
        Targets: encoded as 3, 6, 9,... (y%3==0)
        Heads: encoded as 2, 5, 8,... (z%3==2)
        all greater than 0

        Args:
            hard_fill: whether to hard fill the grid
        Returns:
            None
        """
        if not hard_fill:
            assert self.filled, "Cannot populate the grid if the board is not filled!"
            assert len(self.paths) == self.num_agents, "Something's wrong. Number of paths don't match the wires"

        self.shuffle_all()
        for k, path in enumerate(self.paths):
            head = 3*(k+1)-1
            target = 3*(k+1)
            route = 3*(k+1)-2
            self.grid.fill_grid(path, str_num=head, fill_num=route, end_num=target)

    def remove_routes(self, input_board: Optional[np.ndarray] = None) -> np.ndarray:
        """Removes the routes from the board.
        Args:
            input_board: the board to remove the routes from
        Returns:
            np.array: the board with the routes removed
        """
        if input_board is None:
            self.empty_board = np.where(self.solved_board % 3 != 1, self.solved_board, 0)
            return self.empty_board
        else:
            return np.where(input_board % 3 != 1, input_board, 0)

    def append_all(self, start: Tuple[int, int], end: Tuple[int, int], path: List[Tuple[int, int]]) -> None:
        """Appends the start, end and path to the corresponding lists.
        Args:
            start: start point
            end: end point
            path: path
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
        self._wires_on_board = 0

    def shuffle_all(self) -> None:
        """Shuffles the starts, ends and paths.
        Returns:
            None
        """
        # shuffle all
        zipped = list(zip(self.starts, self.ends, self.paths))
        random.shuffle(zipped)
        self.starts, self.ends, self.paths = map(list, zip(*zipped))

    def remove_wire(self, wire: int) -> None:
        """Removes a wire from the board.
        Args:
            wire: index of the wire to remove
        Returns:
            None
        """
        # remove wire from all lists
        self.starts.pop(wire)
        self.ends.pop(wire)
        self.paths.pop(wire)
        self._wires_on_board -= 1

    @staticmethod
    def count_bends(path: List[tuple]) -> int:
        """Counts the number of bends in a path.
        Args:
            path: the path to count the bends in
        Returns:
            the number of bends in the path
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
            num: number of wires to remove
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
            num: number of wires to remove
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
            num: number of wires to remove
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
                num: number of wires to remove
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
            k: number of bends
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
            k: number of wires to remove
        Returns:
            None
        """
        # Check that there are enough wires to remove
        extra_wires = min(k, len(self.paths))
        for _ in range(extra_wires):
            self.grid.remove_path(self.paths[0])
            self.remove_wire(0)

    def get_clip_method_dict(self) -> Dict[str, Callable]:
        """Creates a dictionary of the methods.
        Returns:
            a dictionary of methods
        """
        return dict(min_bends=self.remove_k_bends, shortest=self.remove_shortest, longest=self.remove_longest,
                    random=self.remove_random, bends=self.remove_fewest_bends, fifo=self.remove_first_k)

    def clip(self, method: str, num: int) -> None:
        """Removes wires using the given method.
        Args:
            method: the method to use
            num: the number of wires to remove
        Returns:
            None
        """
        try:
            clip_func = self.clip_method_dict[method]
        except KeyError:
            raise ValueError(f"Clip Method {method} not found.")

        # Remove the wires using the given method
        clip_func(num)

    def pick_and_place(self, i: int = 0, verbose: Optional[bool] = False) -> Tuple[bool, int]:
        """Picks a random start and end point and tries to place a wire.
        Args:
            i: index of the wire to place
            verbose: whether to print the progress
        Returns:
            a tuple containing if the wire was successfully placed, the path and the number of filled wires
        """
        attempts = 0
        found = False

        while not found and attempts < self.max_attempts:
            if verbose:
                print(f"Fitting wire {i + 1} - attempt {attempts + 1}")
                # Create To Do
                # TODO: Change print to logging
            attempts += 1
            # first pick random start and end
            start, end = self.pick_start_end()
            # then try to fill board
            found, path, _ = self.find_path(start, end)

            if found:
                # a path has been successfully found
                self.append_all(start, end, path)
                self.grid.fill_grid(path)
                self._wires_on_board += 1
                # print(f"Wire {i + 1} placed successfully after {attempts} attempts.")

        return found, self._wires_on_board

    def place_wires(self, verbose: Optional[bool] = False) -> None:
        """Places wires on the board.
        Args:
            verbose: whether to print the progress
        Returns:
            None
        """
        i = self._wires_on_board
        found = True

        while i < self.num_agents and found:
            found, _ = self.pick_and_place(i, verbose=verbose)
            i += 1

    def announce_failure(self):
        """Announces that board unable to be filled after max attempts."""
        print(
            f'Fill unsuccessful for wire {self._wires_on_board + 1} after {self.max_attempts} attempts. '
            f'\n Returning {self.rows}x{self.columns} board with {self._wires_on_board} wire(s).')

    def announce_success(self):
        """Announces that board has been filled."""
        print(f'Fill successful. Returning {self.rows}x{self.columns} board with {self._wires_on_board} wire(s)')

    def return_boards(self, verbose: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray, int]:
        """Returns the board and the solved board.
        Returns:
            tuple containing the board, the solved board
        """
        assert self._wires_on_board <= self.num_agents, "Too many wires have been placed."
        if self._wires_on_board < self.num_agents:
            if verbose:
                self.announce_failure()
            self.populate_grid(hard_fill=True)
            self.partial_board = self.grid.layout.copy()
            self.unsolved_board = self.remove_routes(self.partial_board)
            return self.partial_board, self.unsolved_board, self._wires_on_board
        else:
            if verbose:
                self.announce_success()
            self.filled = True
            self.populate_grid()
            self.solved_board = self.grid.layout.copy()
            self.remove_routes()
            return self.solved_board, self.empty_board, self.num_agents

    def fill_board_with_clipping(self, num_clips: int = 2, method: str = 'bends', verbose: Optional[bool] = False) -> \
            Tuple[np.ndarray, np.ndarray, int]:
        """Fills the board, clips the wires and then tries to fill again.
        Args:
            num_clips: number of wires to remove
            method: method to remove the wires
            verbose: if True, prints the progress
        Returns:
            tuple containing the board, the solved board and the number of filled wires
        """
        self.place_wires(verbose=verbose)
        self.clip(method, num_clips)

        # Reset the grid layout and try to fill again
        self.grid.reset_maze()
        self.place_wires(verbose=verbose)

        return self.return_boards(verbose=verbose)

    def return_training_board(self) -> np.ndarray:
        """Returns the empty board with heads and targets but not wires.
        Returns:
            the empty board
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
            num_clips: number of wires to remove
            methods: method to remove the wires
            num_loops: number of loops to perform
            verbose: if True, prints the progress
        Returns:
            tuple containing the board, the solved board and the number of filled wires
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
            self.grid.reset_maze()

        # Perform the final fill
        self.place_wires(verbose=verbose)

        # Return the boards
        return self.return_boards(verbose=verbose)

    def fill_clip_with_thresholds(self, num_clips: Union[int, List[int]], methods: Union[str, List[str]],
                                  num_loops: Optional[int] = 1, verbose: Optional[bool] = False,
                                  threshold_dict: Dict[str, int] = {
                                      'min_bends', 2}) -> \
            Tuple[np.ndarray, np.ndarray, int]:
        """ Performs a number of fill, clip, fill loops.
        Args:
            num_clips: number of wires to remove
            methods: method to remove the wires
            num_loops: number of loops to perform
            verbose: if True, prints the progress
            threshold_dict: dictionary containing the thresholds for the different metrics


        Returns:
            tuple containing the board, the solved board and the number of filled wires
        """
        if isinstance(num_clips, int):
            num_clips = [num_clips] * num_loops
        if isinstance(methods, str):
            methods = [methods] * num_loops
        assert len(num_clips) == len(methods), "Number of clips and methods must be the same."
        # update num_loops
        num_loops = len(num_clips)
        if verbose:
            print(f"Performing {num_loops} loops of fill-clip-fill.")
        for i in range(num_loops):
            if verbose:
                print(f"Loop {i + 1} of {num_loops}.")
            self.place_wires(verbose=verbose)
            if i > 0:
                threshold_met = self.check_threshold(threshold_dict)
                if threshold_met:
                    break
            self.clip(methods[i], num_clips[i])
            self.grid.reset_maze()
            # Perform the final fill
        self.place_wires(verbose=verbose)
        # Perform final check
        threshold_met = self.check_threshold(threshold_dict)
        if not threshold_met and verbose:
            print(f"Threshold not met for {self.count_non_threshold_paths(threshold_dict, verbose)} path(s).")

        return self.return_boards(verbose=verbose)

    def check_min_bends(self, min_bends: int) -> bool:
        """ Returns the number of bends in the path with the fewest bends
        Args:
            min_bends: minimum number of bends"""
        min_bends_met = True
        for path in self.paths:
            if self.count_bends(path) < min_bends:
                min_bends_met = False
                break
        return min_bends_met

    def check_min_length(self, min_length: int) -> bool:
        """ Returns the number of bends in the path with the fewest bends
        Args:
            min_length: minimum length of the path"""
        min_length_met = True
        for path in self.paths:
            if len(path) < min_length:
                min_length_met = False
                break
        return min_length_met

    def check_threshold(self, threshold_dict: Dict[str, int]) -> bool:
        """ Checks if the thresholds are met"""
        if len(self.paths) < self.num_agents:
            return False
        threshold_met = True
        for metric, threshold in threshold_dict.items():
            if metric == 'min_bends':
                if not self.check_min_bends(threshold):
                    threshold_met = False
                    break
            elif metric == 'min_length':
                if not self.check_min_length(threshold):
                    threshold_met = False
                    break
            else:
                raise ValueError(f"Metric {metric} not recognised.")

        return threshold_met

    def count_non_threshold_paths(self, threshold_dict: Dict[str, int], verbose: Optional[bool] = False) -> int:
        """ Counts the number of paths that do not meet the thresholds
        Args:
            threshold_dict: dictionary containing the thresholds for the different metrics
            """
        non_threshold_paths = 0
        for path in self.paths:
            for key in threshold_dict.keys():

                if key == 'min_length':
                    if len(path) < threshold_dict[key]:
                        if verbose:
                            print(f"Path of length {len(path)} does not meet the threshold.")
                        non_threshold_paths += 1
                        break

                elif key == 'min_bends':
                    if self.count_bends(path) < threshold_dict[key]:
                        if verbose:
                            print(f"Path with {self.count_bends(path)} bend(s) does not meet the threshold.")
                        non_threshold_paths += 1
                        break
                else:
                    raise ValueError(f"Metric {key} not recognised.")
        return non_threshold_paths


if __name__ == '__main__':
    # Example usage
    # Generate a board with 10 rows, 10 columns, 10 wires (num_agents) and with max 10 attempts to place each wire
    board = BFSBoard(rows=10, columns=10, num_agents=10, max_attempts=10)

    # Perform a standard fill
    board.fill_board(verbose=False)
    print(board.return_solved_board())
    print(board.return_training_board())

    # Reset the board
    board.reset_board()
    # Fill the board with 2 wires removed using the 'min_bends' method
    board.fill_board_with_clipping(2, 'min_bends', verbose=False)
    print(board.return_solved_board())
    print(board.return_training_board())

    # Reset the board
    board.reset_board()
    # Fill the board with 2 wires removed using the 'min_bends' method and 2 wires removed using the 'random' method
    board.fill_clip_fill([2, 2], ['min_bends', 'random'], num_loops=2, verbose=False)
    print(board.return_solved_board())
    print(board.return_training_board())

    # Reset the board
    board.reset_board()
    # Test fill_clip_with_thresholds
    test_threshold_dict = {'min_bends': 2, 'min_length': 5}
    clip_nums = [2, 2] * 10
    clip_methods = ['fifo', 'min_bends'] * 10

    board.fill_clip_with_thresholds(clip_nums, clip_methods, verbose=False, threshold_dict=test_threshold_dict)
    print(board.return_solved_board())
    print(board.return_training_board())
