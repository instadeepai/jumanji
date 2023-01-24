"""Initial thought -
Set board size, number of agents (pairs of head and target)
Initialise an empty board (BOARD).
for k in len(agents):
    1. Initialise a random pair of empty points on BOARD
    2. Generate a maze using a DFS algorithm like -() on the BOARD.
    3. Find the optimal path using a BFS.
    4.  If a path exists:
            Store the path.
        else:
            Return to step 1.

    5. Save the BOARD with the path as a wall.

Once complete, populate the BOARD with optimal paths
    """

from collections import deque
import random
import numpy as np

"""
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.layout = np.zeros((rows, cols), dtype=np.int32)
        print(self.layout)

    def dfs(self, row, col):
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        random.shuffle(directions)
        self.layout[row][col] = 1
        for d in directions:
            new_row, new_col = row + d[0], col + d[1]
            if 0 <= new_row < self.rows and (0 <= new_col < self.cols) and self.layout[new_row][new_col] == 0:
                self.layout[(row + new_row) // 2][(col + new_col) // 2] = 1
                self.dfs(new_row, new_col)

    def generate_maze(self, start_row, start_col):
        self.dfs(start_row, start_col)


# Example usage
layout = Grid(10, 10)
layout.generate_maze(0, 0)
for row in layout.layout:
    print(row)


"""


class Grid:
    def __init__(self, rows, cols, grid=None):
        """"""
        self.rows = rows
        self.cols = cols
        if grid is None:
            self.layout = np.zeros((rows, cols), dtype=np.int32)
        else:
            self.layout = grid
        # assert layout.shape == rows, cols
        self.path = []
        # self.visited = [[False] * cols for _ in range(rows)]
        self.visited = np.full((rows, cols), False)

    def bfs(self, start_row, start_col, end_row, end_col):
        queue = deque()
        queue.append((start_row, start_col))
        self.visited[start_row][start_col] = True

        while queue:
            row, col = queue.popleft()
            if row == end_row and col == end_col:
                return True
            for r, c in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                if (0 <= r < self.rows and 0 <= c < self.cols and
                        not self.visited[r][c] and self.layout[r][c]):
                    self.visited[r][c] = True
                    queue.append((r, c))
                    self.path.append((r, c))
        return False

    def solve_maze(self, start_row, start_col, end_row, end_col):
        if self.bfs(start_row, start_col, end_row, end_col):
            return self.path
        else:
            return None

    def bfs_maze(self, start, end):
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

            # Loop through possible movements
            for i in range(4):
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

        # If we have not found a path
        return False, [], 0

    def fill_grid(self, path, str_num=None, fill_num=None, end_num=None):
        if str_num is None:
            str_num = end_num = fill_num = 1
        for i, pt in enumerate(path):
            if i == 0:
                self.layout[pt[0]][pt[1]] = str_num
            elif i == len(path) - 1:
                self.layout[pt[0]][pt[1]] = end_num
            else:
                self.layout[pt[0]][pt[1]] = fill_num


"""
# Example usage
grid = np.array([[1, 0, 1, 1, 1],
                 [1, 0, 1, 0, 1],
                 [1, 0, 1, 0, 1],
                 [1, 0, 1, 0, 1],
                 [1, 1, 1, 0, 1]])

solver = Grid(5, 5, grid)
# path = solver.solve_maze(0, 0, 4, 4)
_,path,_ = solver.bfs_maze((0, 0), (4, 4))
print(path)
if not path:
    print("No solution found.")
else:
    print("Solution path:", path)

print(grid)
for pt in path:
    grid[pt[0]][pt[1]] = 2

print(grid)

"""


class Board:
    def __init__(self, rows, cols, wires, max_attempts=10):

        self.rows = rows
        self.cols = cols
        self.grid = Grid(rows, cols)
        self.wires = wires
        self.paths = []
        self.starts = []
        self.ends = []
        self.filled = False
        self.filled_board = None
        self.max_attempts = max_attempts

    def pick_start_end(self):
        options = np.argwhere(self.grid.layout == 0).tolist()
        points = random.sample(options, 2)
        # start, end = points[0], points[1]
        return tuple(points[0]), tuple(points[1])

    def place_points(self, start, end):
        found, path, num = self.grid.bfs_maze(start, end)
        return found, path, num

    def fill_board(self):
        for i in range(self.wires):

            found = False
            attempts = 0

            while not found and attempts < self.max_attempts:
                print(f"Fitting {i + 1} wire - attempt {attempts + 1}")
                attempts += 1
                # first pick random start and end
                start, end = self.pick_start_end()
                # then try to fill board
                found, path, _ = self.place_points(start, end)

                if found:
                    # a path has been successfully found
                    self.append_all(start, end, path)
                    self.grid.fill_grid(path)
            if not found:
                self.reset_board()
                print(f'Fill unsuccessful for wire {i} after {self.max_attempts} attempts')
                return None
        # Having successfully filled board, we now populate
        self.filled = True
        self.populate_grid()
        self.filled_board = self.grid.layout
        return self.filled_board

    def populate_grid(self):
        # Heads: encoded as 4, 7 , 10, ... (x%3==1)
        #    Targets: encoded as 3, 6, 9,... (y%3==0)
        #    Routes: encoded as 2, 5, 8,... (z%3==0) 
        assert self.filled, "Cannot populate the grid if the board is not filled!"
        assert len(self.paths) == self.wires, "Something's wrong. Number of paths don't match the wires"
        for k, path in enumerate(self.paths):
            head = 3 * (k + 1) + 1
            target = 3 * (k + 1)
            route = 3 * (k + 1) - 1
            self.grid.fill_grid(path, str_num=head, fill_num=route, end_num=target)

    def append_all(self, start, end, path):
        self.starts.append(start)
        self.ends.append(end)
        self.paths.append(path)

    def reset_board(self):
        # reset grid, starts, ends and paths
        self.grid = Grid(self.rows, self.cols)
        self.paths = []
        self.starts = []
        self.ends = []


# Worked Example
test_board = Board(20, 20, 12)
# print(test_board.grid.layout)

test_board.fill_board()
print(test_board.filled_board)
