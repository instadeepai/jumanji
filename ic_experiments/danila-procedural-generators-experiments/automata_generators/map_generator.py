# Copyright 2021 - M. Pecheux
# [MapGenerator] A procedural map generation based on a cellular automaton
# ------------------------------------------------------------------------------
# map_generator.py - TileType and Map generator classes
# ==============================================================================
import sys
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation


class TileType(object):
    
    """Util class to represent a tile type (with specific color and smoothing
    transformations)."""

    def __init__(self, color, smoothing_threshold, smoothing_transform):
        self.color               = color
        self.smoothing_threshold = smoothing_threshold
        self.smoothing_transform = smoothing_transform

        if isinstance(self.smoothing_threshold, list) and \
            not isinstance(self.smoothing_transform, list):
            print('[Error] Cannot initialize the tile type.\nYou cannot set a '
                  'tile type smoothing with one threshold and multiple transform'
                  'results.')
            sys.exit(1)
        if isinstance(self.smoothing_transform, list) and \
            not isinstance(self.smoothing_threshold, list):
            print('[Error] Cannot initialize the tile type.\nYou cannot set a '
                  'tile type smoothing with multiple thresholds and one '
                  'transform result.')
            sys.exit(1)

        if not (isinstance(self.smoothing_threshold, float) or \
            isinstance(self.smoothing_threshold, list)):
            print('[Error] Cannot initialize the tile type.\nA tile type '
                  'smoothing must use either a single threshold and result, or '
                  'a list of threshold and results.')
            sys.exit(1)

    def apply_smoothing(self, same_num_neighbors, max_num_neighbors):
        if isinstance(self.smoothing_threshold, float):
            t  = int(max_num_neighbors * self.smoothing_threshold)
            if same_num_neighbors < t: return self.smoothing_transform
        elif isinstance(self.smoothing_threshold, list):
            t0 = int(max_num_neighbors * self.smoothing_threshold[0])
            t1 = int(max_num_neighbors * self.smoothing_threshold[1])
            if same_num_neighbors < t0: return self.smoothing_transform[0]
            if same_num_neighbors > t1: return -1


class MapGenerator(object):

    """Map generator based on the cellular automaton paradigm: a random set of
    tiles is taken as initialization, then iterations of various smoothing
    methods transform the map to create more realistic areas.

    Parameters
    ----------
    width : int
        Horizontal size of the map to generate.
    height : int
        Vertical size of the map to generate.
    """

    def __init__(self, width, height, **kwargs):
        self.width        = width
        self.height       = height
        self.display_size = kwargs.get('display_size', [8, 8])

        self.tile_types   = kwargs.get('tile_types', MapGenerator._dft_tile_types())
        self.tiles        = np.empty((height, width), dtype=np.int32)

    @staticmethod
    def _dft_tile_types():
        return [
            (30, TileType((0.5, 0.9, 0.3), 0.2, 1)),
            (20, TileType((0.15, 0.6, 0.2), 0.1, 0)),
            (20, TileType((0.65, 0.65, 0.1), [0.1, 0.6], [0, 2])),
            (30, TileType((0.0, 0.05, 0.75), [0.1, 0.42], [0, 3])),
        ]

    def _is_in_map(self, x, y):
        return (x >= 0 and x < self.width) and (y >= 0 and y < self.height)

    def _num_same_neighbors(self, x, y, size, ref_type=None):
        same_neighbors = 0
        if ref_type is None: ref_type = self.tiles[y, x]
        for i in range(x - size, x + size + 1):
            for j in range(y - size, y + size + 1):
                if i == x and j == y: continue
                if self._is_in_map(i, j) and self.tiles[j, i] == ref_type:
                    same_neighbors += 1
        return same_neighbors

    def initialize(self, seed):
        """Initializes a map from the given seed.

        Parameters
        ----------
        seed : int
            Seed to initialize pseudo-random generators.
        """
        # initialize pseudo-random generators
        random.seed(seed)
        # create map
        for y in range(self.height):
            for x in range(self.width):
                r = int(random.random() * 100.)
                t = len(self.tile_types) - 1
                thresh = 0
                for i, tile_type in enumerate(self.tile_types):
                    threshold, _ = tile_type
                    thresh += threshold
                    if r < thresh:
                        t = i
                        break
                self.tiles[y, x] = t

    def smooth(self, smooth_size):
        """Smooths the map depending on the type of the neighboring tiles.

        Parameters
        ----------
        smooth_size : int
            Maximum distance for the neighbors to check from the current tile.
        """
        # get maximum number of neighbors to check for
        num_neighbors = sum((8 * (i+1) for i in range(smooth_size)))
        half_s = max(1, smooth_size // 2)

        # go through the map and apply smoothing rules
        new_tiles = np.copy(self.tiles)
        for y in range(self.height):
            for x in range(self.width):
                # get number of neighbors with the same tile type
                n     = self._num_same_neighbors(x, y, smooth_size)
                # apply rule depending on tile type
                t     = self.tile_types[self.tiles[y, x]][1]
                new_t = t.apply_smoothing(n, num_neighbors)
                if new_t is not None:
                    # current tile modification
                    if new_t >= 0: new_tiles[y, x] = new_t
                    # neighbor tiles modification
                    else:
                        ref_t = self.tiles[y, x]
                        for yy in range(y - half_s, y + half_s + 1):
                            for xx in range(x - half_s, x + half_s + 1):
                                if self._is_in_map(xx, yy):
                                    new_tiles[yy, xx] = ref_t
        self.tiles = new_tiles

    def earthify(self):
        """Removes the water tiles in the middle of continents."""
        # go through the map and only get water tiles
        new_tiles = np.copy(self.tiles)
        for y in range(self.height):
            for x in range(self.width):
                if self.tiles[y, x] == len(self.tile_types) - 1:
                    # get number of neighbors with the same tile type
                    n = self._num_same_neighbors(x, y, 2)
                    # transform tiles with too little water neighbors
                    # (if the tile is close to a border, the threshold is lower)
                    if x <= 1 or x >= self.width - 1 or y <= 1 or y >= self.height - 1:
                        if n < 12: new_tiles[y, x] = 0
                    elif n < 18: new_tiles[y, x] = 0

        self.tiles = new_tiles

    def clean(self):
        """Removes the water tiles in the middle of continents."""
        # go through the map and only get water tiles
        ref_type = len(self.tile_types) - 1
        for y in range(self.height):
            for x in range(self.width):
                if self.tiles[y, x] == ref_type:
                    # get number of neighbors with the same tile type
                    n = self._num_same_neighbors(x, y, 2)
                    if n < 12: self.tiles[y, x] = np.random.randint(2)
                else:
                    # get number of neighbors with water tile type
                    n = self._num_same_neighbors(x, y, 2, ref_type)
                    if n > 22: self.tiles[y, x] = ref_type

    def init_output(self):
        """Initializes all the useful variables to output the map."""
        self.fig = plt.figure(figsize=self.display_size)
        self.ax  = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')

        self.cmap = LinearSegmentedColormap.from_list(
            'map_colors',
            [t.color for _, t in self.tile_types],
            N=len(self.tile_types)
        )

    def output(self, label=None):
        """Outputs the map."""
        artists = [plt.imshow(self.tiles, cmap=self.cmap)]
        if label is not None:
            if not isinstance(label, str): label = str(label)
            artists.append(self.ax.text(
                self.width - self.width//20,
                self.height - self.height//20,
                label, size=12, ha='right', bbox=dict(fc='white')))
        return artists

    @staticmethod
    def run(**kwargs):
        """Runs a MapGenerator to create a random map."""
        seed            = kwargs.pop('seed', int(100. * np.random.random()))
        smooth_size     = kwargs.pop('smooth_size', [4, 2, 6])
        earthifications = kwargs.pop('earthifications', 2)
        clean_up        = kwargs.pop('clean_up', 1)
        with_animation  = kwargs.pop('with_animation', False) # if True, output
                                                              # intermediate results
        export_file     = kwargs.pop('export_file', None)

        # create model instance
        model = MapGenerator(**kwargs)

        # initialize map
        print('Initializing the map.\n' + '=' * 21 + '\n')
        model.initialize(seed)
        # initialize the output
        model.init_output()
        ims = []

        # smooth map
        if isinstance(smooth_size, int):
            smooth_amount = kwargs.pop('smooth_amount', 3)
            for i in range(smooth_amount):
                debug_str = 'Smooth #{} (size = {})'.format(i+1, smooth_size)
                print('\t{}\n\t'.format(debug_str) + '-' * len(debug_str))
                if with_animation:
                    ims.append(model.output(debug_str))
                model.smooth(smooth_size)
        elif isinstance(smooth_size, list):
            if 'smooth_amount' in kwargs:
                print('[Warning] A specific list of smoothing sizes is provided: '
                      ' this overrides the "smooth_amount" parameter.')
            for i, s in enumerate(smooth_size):
                if s > 0:
                    debug_str = 'Smooth #{} (size = {})'.format(i+1, s)
                    print('\t{}\n\t'.format(debug_str) + '-' * len(debug_str))
                    if with_animation:
                        ims.append(model.output(debug_str))
                    model.smooth(s)
        # "earthify", ie remove water from the continents
        for i in range(earthifications):
            debug_str = '"Earthification" #{}'.format(i+1)
            print('\t{}\n\t'.format(debug_str) + '-' * len(debug_str))
            model.earthify()
            if with_animation: ims.append(model.output(debug_str))
        # cleanup, ie soften areas
        for i in range(clean_up):
            debug_str = 'Clean up #{}'.format(i+1)
            print('\t{}\n\t'.format(debug_str) + '-' * len(debug_str))
            model.clean()
            if with_animation: ims.append(model.output(debug_str))

        if with_animation:
            # add multiple images (arbitrary number) to stay still
            # on the final frame
            for _ in range(5):
                ims.append(model.output())

        # prepare visual output
        print('\nExporting and outputting the map.\n' + '=' * 33)
        # export result
        if export_file is not None:
            extension = export_file.split('.')[-1].lower()
            if with_animation and not extension == 'gif':
                print('[Warning] You can only export animated maps with the '
                      '"gif" extension. The map was not exported.')
            elif not with_animation and not extension in ['png', 'jpg', 'jpeg']:
                print('[Warning] You can only export non-animated maps with the '
                      '"png", "jpg" or "jpeg" extensions. The map was not exported.')
            else:
                print('Exporting to: "{}"\n'.format(export_file))
                if with_animation:
                    output = animation.ArtistAnimation(model.fig, ims, interval=500, blit=False)
                    output.save(export_file, writer='imagemagick')
                else:
                    model.output()
                    model.fig.savefig(export_file)
                plt.close(model.fig)
        # output result
        else:
            if with_animation:
                output = animation.ArtistAnimation(model.fig, ims, interval=500, blit=False)
            else:
                model.output()
            plt.show()
