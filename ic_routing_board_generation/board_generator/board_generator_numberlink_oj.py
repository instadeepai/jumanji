"""
Note: This is a slightly modified version of the gen folder from:
https://github.com/thomasahle/numberlink

TODO: Properly reference!!
"""


"""
Copying stuff from gen.py
"""

import sys
import random
import collections
from copy import deepcopy
import itertools
import argparse
import numpy as np
import sys
import collections
import string
from colorama import Fore, Style
from colorama import init as init_colorama

from ic_routing_board_generation.board_generator.abstract_board import AbstractBoard


# Number of tries at adding loops to the grid before redrawing the side paths.
LOOP_TRIES = 1000


"""
Functions from mitm
"""
import random
from collections import Counter, defaultdict
import itertools

# Un
#T, L, R = range(3)


class Path:
    def __init__(self, steps):
        self.steps = steps
        self.T, self.L, self.R = 0, 1, 2

    def xys(self, dx=0, dy=1):
        """ Yields all positions on path """
        x, y = 0, 0
        yield (x, y)
        for step in self.steps:
            x, y = x + dx, y + dy
            yield (x, y)
            if step == self.L:
                dx, dy = -dy, dx
            if step == self.R:
                dx, dy = dy, -dx
            elif step == self.T:
                x, y = x + dx, y + dy
                yield (x, y)

    def test(self):
        """ Tests path is non-overlapping. """
        ps = list(self.xys())
        return len(set(ps)) == len(ps)

    def test_loop(self):
        """ Tests path is non-overlapping, except for first and last. """
        ps = list(self.xys())
        seen = set(ps)
        return len(ps) == len(seen) or len(ps) == len(seen) + 1 and ps[0] == ps[-1]

    def winding(self):
        return self.steps.count(self.R) - self.steps.count(self.L)

    def __repr__(self):
        """ Path to string """
        return ''.join({self.T: '2', self.R: 'R', self.L: 'L'}[x] for x in self.steps)

    def show(self):
        import matplotlib.pyplot as plt
        xs, ys = zip(*self.xys())
        plt.plot(xs, ys)
        plt.axis('scaled')
        plt.show()


def unrotate(x, y, dx, dy):
    """ Inverse rotate x, y by (dx,dy), where dx,dy=0,1 means 0 degrees.
        Basically rotate(dx,dy, dx,dy) = (0, 1). """
    while (dx, dy) != (0, 1):
        x, y, dx, dy = -y, x, -dy, dx
    return x, y


class Mitm:
    def __init__(self, lr_price, t_price):
        self.lr_price = lr_price
        self.t_price = t_price
        self.inv = defaultdict(list)
        self.list = []
        self.T, self.L, self.R = 0, 1, 2

    def prepare(self, budget):
        dx0, dy0 = 0, 1
        for path, (x, y, dx, dy) in self._good_paths(0, 0, dx0, dy0, budget):
            self.list.append((path, x, y, dx, dy))
            self.inv[x, y, dx, dy].append(path)

    def rand_path(self, xn, yn, dxn, dyn):
        """ Returns a path, starting at (0,0) with dx,dy = (0,1)
            and ending at (xn,yn) with direction (dxn, dyn) """
        while True:
            path, x, y, dx, dy = random.choice(self.list)
            path2s = self._lookup(dx, dy, xn - x, yn - y, dxn, dyn)
            if path2s:
                path2 = random.choice(path2s)
                joined = Path(path + path2)
                if joined.test():
                    return joined

    def rand_path2(self, xn, yn, dxn, dyn):
        """ Like rand_path, but uses a combination of a fresh random walk and
            the lookup table. This allows for even longer paths. """
        seen = set()
        path = []
        while True:
            seen.clear()
            del path[:]
            x, y, dx, dy = 0, 0, 0, 1
            seen.add((x, y))
            for _ in range(2 * (abs(xn) + abs(yn))):
                # We sample with weights proportional to what they are in _good_paths()
                step, = random.choices(
                    [self.L, self.R, self.T], [1 / self.lr_price, 1 / self.lr_price, 2 / self.t_price])
                path.append(step)
                x, y = x + dx, y + dy
                if (x, y) in seen:
                    break
                seen.add((x, y))
                if step == self.L:
                    dx, dy = -dy, dx
                if step == self.R:
                    dx, dy = dy, -dx
                elif step == self.T:
                    x, y = x + dx, y + dy
                    if (x, y) in seen:
                        break
                    seen.add((x, y))
                if (x, y) == (xn, yn):
                    return Path(path)
                ends = self._lookup(dx, dy, xn - x, yn - y, dxn, dyn)
                if ends:
                    return Path(tuple(path) + random.choice(ends))

    def rand_loop(self, clock=0):
        """ Set clock = 1 for clockwise, -1 for anti clockwise. 0 for don't care. """
        while True:
            # The list only contains 0,1 starting directions
            path, x, y, dx, dy = random.choice(self.list)
            # Look for paths ending with the same direction
            path2s = self._lookup(dx, dy, -x, -y, 0, 1)
            if path2s:
                path2 = random.choice(path2s)
                joined = Path(path + path2)
                # A clockwise path has 4 R's more than L's.
                if clock and joined.winding() != clock * 4:
                    continue
                if joined.test_loop():
                    return joined

    def _good_paths(self, x, y, dx, dy, budget, seen=None):
        if seen is None:
            seen = set()
        if budget >= 0:
            yield (), (x, y, dx, dy)
        if budget <= 0:
            return
        seen.add((x, y))  # Remember cleaning this up (A)
        x1, y1 = x + dx, y + dy
        if (x1, y1) not in seen:
            for path, end in self._good_paths(
                    x1, y1, -dy, dx, budget - self.lr_price, seen):
                yield (self.L,) + path, end
            for path, end in self._good_paths(
                    x1, y1, dy, -dx, budget - self.lr_price, seen):
                yield (self.R,) + path, end
            seen.add((x1, y1))  # Remember cleaning this up (B)
            x2, y2 = x1 + dx, y1 + dy
            if (x2, y2) not in seen:
                for path, end in self._good_paths(
                        x2, y2, dx, dy, budget - self.t_price, seen):
                    yield (self.T,) + path, end
            seen.remove((x1, y1))  # Clean up (B)
        seen.remove((x, y))  # Clean up (A)

    def _lookup(self, dx, dy, xn, yn, dxn, dyn):
        """ Return cached paths coming out of (0,0) with direction (dx,dy)
            and ending up in (xn,yn) with direction (dxn,dyn). """
        # Give me a path, pointing in direction (0,1) such that when I rotate
        # it to (dx, dy) it ends at xn, yn in direction dxn, dyn.
        xt, yt = unrotate(xn, yn, dx, dy)
        dxt, dyt = unrotate(dxn, dyn, dx, dy)
        return self.inv[xt, yt, dxt, dyt]


"""
Functions from grid.py
"""
def sign(x):
    if x == 0:
        return x
    return -1 if x < 0 else 1


class UnionFind:
    def __init__(self, initial=None):
        self.uf = initial or {}

    def union(self, a, b):
        a_par, b_par = self.find(a), self.find(b)
        self.uf[a_par] = b_par

    def find(self, a):
        if self.uf.get(a, a) == a:
            return a
        par = self.find(self.uf.get(a, a))
        # Path compression
        self.uf[a] = par
        return par


class Grid:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.grid = {}

    def __setitem__(self, key, val):
        self.grid[key] = val

    def __getitem__(self, key):
        return self.grid.get(key, ' ')

    def __repr__(self):
        res = []
        for y in range(self.h):
            res.append(''.join(self[x, y] for x in range(self.w)))
        return '\n'.join(res)

    def __iter__(self):
        return iter(self.grid.items())

    def __contains__(self, key):
        return key in self.grid

    def __delitem__(self, key):
        del self.grid[key]

    def clear(self):
        self.grid.clear()

    def values(self):
        return self.grid.values()

    def shrink(self):
        """ Returns a new grid of half the height and width """
        small_grid = Grid(self.w // 2, self.h // 2)
        for y in range(self.h // 2):
            for x in range(self.w // 2):
                small_grid[x, y] = self[2 * x + 1, 2 * y + 1]
        return small_grid

    def test_path(self, path, x0, y0, dx0=0, dy0=1):
        """ Test whether the path is safe to draw on the grid, starting at x0, y0 """
        return all(0 <= x0 - x + y < self.w and 0 <= y0 + x + y < self.h
                and (x0 - x + y, y0 + x + y) not in self for x, y in path.xys(dx0, dy0))

    def draw_path(self, path, x0, y0, dx0=0, dy0=1, loop=False):
        """ Draws path on the grid. Asserts this is safe (no overlaps).
            For non-loops, the first and the last character is not drawn,
            as we don't know what shape they should have. """
        ps = list(path.xys(dx0, dy0))
        # For loops, add the second character, so we get all rotational tripples:
        # abcda  ->  abcdab  ->  abc, bcd, cda, dab
        if loop:
            assert ps[0] == ps[-1], (path, ps)
            ps.append(ps[1])
        for i in range(1, len(ps) - 1):
            xp, yp = ps[i - 1]
            x, y = ps[i]
            xn, yn = ps[i + 1]
            self[x0 - x + y, y0 + x + y] = {
                (1, 1, 1): '<', (-1, -1, -1): '<',
                (1, 1, -1): '>', (-1, -1, 1): '>',
                (-1, 1, 1): 'v', (1, -1, -1): 'v',
                (-1, 1, -1): '^', (1, -1, 1): '^',
                (0, 2, 0): '\\', (0, -2, 0): '\\',
                (2, 0, 0): '/', (-2, 0, 0): '/'
            }[xn - xp, yn - yp, sign((x - xp) * (yn - y) - (xn - x) * (y - yp))]

    def make_tubes(self):
        uf = UnionFind()
        tube_grid = Grid(self.w, self.h)
        for x in range(self.w):
            d = '-'
            for y in range(self.h):
                # We union things down and to the right.
                # This means ┌ gets to union twice.
                for dx, dy in {
                        '/-': [(0, 1)], '\\-': [(1, 0), (0, 1)],
                        '/|': [(1, 0)],
                        ' -': [(1, 0)], ' |': [(0, 1)],
                        'v|': [(0, 1)], '>|': [(1, 0)],
                        'v-': [(0, 1)], '>-': [(1, 0)],
                }.get(self[x, y] + d, []):
                    uf.union((x, y), (x + dx, y + dy))
                # We change alll <>v^ to x.
                tube_grid[x, y] = {
                    '/-': '┐', '\\-': '┌',
                    '/|': '└', '\\|': '┘',
                    ' -': '-', ' |': '|',
                }.get(self[x, y] + d, 'x')
                # We change direction on v and ^, but not on < and >.
                if self[x, y] in '\\/v^':
                    d = '|' if d == '-' else '-'
        return tube_grid, uf

    def clear_path(self, path, x, y):
        """ Removes everything contained in the path (loop) placed at x, y. """
        path_grid = Grid(self.w, self.h)
        path_grid.draw_path(path, x, y, loop=True)
        for key, val in path_grid.make_tubes()[0]:
            if val == '|':
                self.grid.pop(key, None)

"""
Functions from draw.py
"""


def color_tubes(grid, no_colors=False):
    """ Add colors and numbers for drawing the grid to the terminal. """
    colors = ['']
    reset = ''
    tube_grid, uf = grid.make_tubes()
    # Change this to string.digits to get numbers instead of letters.
    letters = [str(i) for i in range(1, 1000)]

    char = collections.defaultdict(lambda: letters[len(char)])
    col = collections.defaultdict(lambda: colors[len(col) % len(colors)])
    for x in range(tube_grid.w):
        for y in range(tube_grid.h):
            if tube_grid[x, y] == 'x':
                tube_grid[x, y] = char[uf.find( (x, y))]
            tube_grid[x, y] = col[uf.find( (x, y))] + tube_grid[x, y] + reset
    return tube_grid, char

"""
main from gen!
"""
class NumberLinkBoard(AbstractBoard):
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.no_colors = True
        self.zero = True
        self.solve = False
        self.no_pipes = True
        self.terminal_only = True
        self.verbose = False


        self.w, self.h = self.width, self.height
        if self.w < 4 or self.h < 4:
            print('Please choose width and height at least 4.')
            return

        self.n = self.num_agents
        self.min_numbers = self.n
        self.max_numbers = self.n

        # Make the board
        self.main()


    def has_loops(self, grid, uf):
        """ Check whether the puzzle has loops not attached to an endpoint. """
        groups = len({uf.find((x, y)) for y in range(grid.h) for x in range(grid.w)})
        ends = sum(bool(grid[x, y] in 'v^<>') for y in range(grid.h) for x in range(grid.w))
        return ends != 2 * groups


    def has_pair(self, tg, uf):
        """ Check for a pair of endpoints next to each other. """
        for y in range(tg.h):
            for x in range(tg.w):
                for dx, dy in ((1, 0), (0, 1)):
                    x1, y1 = x + dx, y + dy
                    if x1 < tg.w and y1 < tg.h:
                        if tg[x, y] == tg[x1, y1] == 'x' \
                                and uf.find( (x, y)) == uf.find( (x1, y1)):
                            return True
        return False


    def has_tripple(self, tg, uf):
        """ Check whether a path has a point with three same-colored neighbours.
            This would mean a path is touching itself, which is generally not
            allowed in pseudo-unique puzzles.
            (Note, this also captures squares.) """
        for y in range(tg.h):
            for x in range(tg.w):
                r = uf.find( (x, y))
                nbs = 0
                for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    x1, y1 = x + dx, y + dy
                    if 0 <= x1 < tg.w and 0 <= y1 < tg.h and uf.find( (x1, y1)) == r:
                        nbs += 1
                if nbs >= 3:
                    return True
        return False


    def test_ready(self):
                # Test if grid is ready to be returned.
                self.sg = self.grid.shrink()
                self.stg, self.uf = self.sg.make_tubes()
                numbers = list(self.stg.values()).count('x') // 2
                return self.min_numbers <= numbers <= self.max_numbers \
                        and not self.has_loops(self.sg, self.uf) \
                        and not self.has_pair(self.stg, self.uf) \
                        and not self.has_tripple(self.stg, self.uf) \

    def make(self, w, h, mitm, min_numbers=0, max_numbers=1000):
        """ Creates a grid of size  w x h  without any loops or squares.
            The mitm table should be genearted outside of make() to give
            the best performance.
            """

        # Internally we work on a double size grid to handle crossings
        self.grid = Grid(2 * w + 1, 2 * h + 1)

        self.gtries = 0
        while True:
            # Previous tries may have drawn stuff on the grid
            self.grid.clear()


            # Add left side path
            path = mitm.rand_path2(h, h, 0, -1)
            if not self.grid.test_path(path, 0, 0):
                continue
            self.grid.draw_path(path, 0, 0)
            # Draw_path doesn't know what to put in the first and last squares
            self.grid[0, 0], self.grid[0, 2 * h] = '\\', '/'
            # Add right side path
            path2 = mitm.rand_path2(h, h, 0, -1)
            if not self.grid.test_path(path2, 2 * w, 2 * h, 0, -1):
                continue
            self.grid.draw_path(path2, 2 * w, 2 * h, 0, -1)
            self.grid[2 * w, 0], self.grid[2 * w, 2 * h] = '/', '\\'

            # The puzzle might already be ready to return
            if self.test_ready():
                return self.grid.shrink()

            # Add loops in the middle
            # Tube version of full grid, using for tracking orientations.
            # This doesn't make so much sense in terms of normal numberlink tubes.
            tg, _ = self.grid.make_tubes()
            # Maximum number of tries before retrying main loop
            for tries in range(LOOP_TRIES):
                x, y = 2 * random.randrange(w), 2 * random.randrange(h)

                # If the square square doen't have an orientation, it's a corner
                # or endpoint, so there's no point trying to add a loop there.
                if tg[x, y] not in '-|':
                    continue

                path = mitm.rand_loop(clock=1 if tg[x, y] == '-' else -1)
                if self.grid.test_path(path, x, y):
                    # A loop may not overlap with anything, and may even have
                    # the right orientation, but if it 'traps' something inside it, that
                    # might now have the wrong orientation.
                    # Hence we clear the insides.
                    self.grid.clear_path(path, x, y)

                    # Add path and recompute orientations
                    self.grid.draw_path(path, x, y, loop=True)
                    tg, _ = self.grid.make_tubes()

                    # Run tests to see if the puzzle is nice
                    sg = self.grid.shrink()
                    stg, uf = sg.make_tubes()
                    numbers = list(stg.values()).count('x') // 2
                    if numbers > max_numbers:
                        self.debug('Exceeded maximum number of number pairs.')
                        break
                    if self.test_ready():
                        self.debug(f'Finished in {tries} tries.')
                        self.debug(f'{numbers} numbers')
                        return sg

            self.debug(self.grid)
            self.debug(f'Gave up after {tries} tries')



    def debug(self, s):
        verbose = False
        try:
            if verbose:
                print(s, file=sys.stderr)
        except NameError:
            pass


    def main(self):
        mitm = Mitm(lr_price=2, t_price=1)
        # Using a larger path length in mitm might increase puzzle complexity, but
        # 8 or 10 appears to be the sweet spot if we want small sizes like 4x4 to
        # work.
        mitm.prepare(min(20, max(self.h, 6)))
        self.debug('Generating puzzle...')

        grid = self.make(self.w, self.h, mitm, self.min_numbers, self.max_numbers)
        tube_grid, uf = grid.make_tubes()
        color_grid, mapping = color_tubes(grid, no_colors=self.no_colors)

        # Print stuff
        self.debug(grid)

        self.training_board = np.zeros((self.w, self.h), dtype=int)
        already_found = dict()
        if self.zero:
            # Print puzzle in 0 format
            for y in range(color_grid.h):
                for x in range(color_grid.w):
                    if grid[x, y] in 'v^<>':
                        #print(int(color_grid[x, y]))
                        if int(color_grid[x,y]) * 3 - 1 in already_found:
                            # If already found, make it a target
                            self.training_board[y,x] = int(color_grid[x,y]) * 3
                        else:
                            self.training_board[y,x] = int(color_grid[x,y]) * 3 - 1
                            #print(already_found)
                            already_found[self.training_board[y,x]] = True 
                        #print(color_grid[x, y], end=' ')
                    else:
                        pass
                        #print('0', end=' ')
                #print()
        self.color_grid = color_grid

    def grid_to_np(self, grid):
        # Create empty numoy array of strs
        np_board = np.empty((self.w, self.h), dtype=object)
        # np_board = ((self.w, self.h), dtype=str)
        for y in range(grid.h):
            for x in range(grid.w):
                    np_board[y,x] = grid[x,y]
        return np_board
    
    def extract_wire(self, start, np_board):
        """
        Given a start and a board, extract the wire
        """
        # Initialise the wire
        wire = []
        # initialise the current position
        current_pos = start
        # Initialise the start direction:
        # This involves checking the neighbours of the start
        # TODO: check these correspond to up / down / left / right
        if np_board[(current_pos[0] + 0, min(current_pos[1] + 1, self.h - 1))] in '-┐┘':
            current_dir = (0,1)
        elif np_board[(current_pos[0] + 0, max(current_pos[1] + -1, 0))] in '-┌└':
            current_dir = (0,-1)
        elif np_board[(max(current_pos[0] - 1, 0), current_pos[1] + 0)] in '|┌┐':
            current_dir = (-1,0)
        elif np_board[(min(current_pos[0] + 1, self.w - 1), current_pos[1] + 0)] in '|└┘':
            current_dir = (1,0)
        
        started = True
        # Now, follow the wire
        while np_board[current_pos[0], current_pos[1]] in '┌┐└┘||-┐' or started == True:
            started = False
            # Add the current position to the wire
            #print("step")
            wire += [current_pos]
            # Follow the wire
            if np_board[current_pos[0], current_pos[1]] == '┌':
                if current_dir == (0,-1):
                    current_dir = (1,0)
                elif current_dir == (-1,0):
                    current_dir = (0,1)
            elif np_board[current_pos[0], current_pos[1]] == '┐':
                if current_dir == (0,1):
                    current_dir = (1,0)
                elif current_dir == (-1,0):
                    current_dir = (0,-1)
            elif np_board[current_pos[0], current_pos[1]] == '└':
                if current_dir == (0,-1):
                    current_dir = (-1,0)
                elif current_dir == (1,0):
                    current_dir = (0,1)
            elif np_board[current_pos[0], current_pos[1]] == '┘':
                if current_dir == (0,1):
                    current_dir = (-1,0)
                elif current_dir == (1,0):
                    current_dir = (0,-1)
            elif np_board[current_pos[0], current_pos[1]] == '│':
                if current_dir == (1,0):
                    current_dir = (-1,0)
                elif current_dir == (-1,0):
                    current_dir = (1,0)
            elif np_board[current_pos[0], current_pos[1]] == '─':
                if current_dir == (0,1):
                    current_dir = (0,-1)
                elif current_dir == (0,-1):
                    current_dir = (0,1)
            # Move to the next position
            current_pos = (current_pos[0] + current_dir[0], current_pos[1] + current_dir[1])
            #print(np_board[current_pos[0], current_pos[1]])
            #print(np_board[current_pos[0], current_pos[1]] in '┌┐└┘||-┐')

        # Return the wire
        return wire[1:]

    
    def return_solved_board(self) -> np.ndarray:
        """
        Has to extract the solved board from the color grid
        Similar method will be used as with WFC
        """
        board = deepcopy(self.color_grid)
        # Convert this to a numpy array
        np_board = self.grid_to_np(board)
        # Initialise the output board
        output_board = np.zeros((self.w, self.h), dtype=int)
        # Continue until all the numbers are found
        for k in range(self.num_agents):
            ends = []
            for i in range(self.width):
                for j in range(self.height):
                    if np_board[i,j] == str(k+1):
                        ends += [(i,j)]
            #print(ends)
            # Extract the wire with this start
            wire = self.extract_wire(ends[0], np_board)
            #print("wire k", wire)
            # Add the wire to the output board
            for pos in wire:
                output_board[pos[0], pos[1]] = 3*(k+1)-2
            # Add the start to the output board
            output_board[ends[0][0], ends[0][1]] = 3*(k+1)-1
            # Add the end to the output board
            output_board[ends[1][0], ends[1][1]] = 3*(k+1)
        
        return output_board

    def return_training_board(self) -> np.ndarray:
        return self.training_board


if __name__ == '__main__':
    board = NumberLinkBoard(8,8,5)
    unsolved = board.return_training_board()
    print(unsolved)
    output = board.return_solved_board()
    print(output)
