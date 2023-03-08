# Copyright 2021 - M. Pecheux
# [MapGenerator] A procedural map generation based on a cellular automaton
# ------------------------------------------------------------------------------
# main.py - A few examples of usage of the MapGenerator class
# ==============================================================================
from map_generator import MapGenerator, TileType

# The different examples will display a Matplotlib figure at the end of
# the process. To continue on with the next one, simply close the figure :)


# Ex 1:
# -----
# Run a Mapper with default parameters
MapGenerator.run(width=300, height=300, smooth_size=[4, 2, 6, 3],
                 earthifications=5,
                 clean_up=4,
                 export_file='rpg-map.jpg')

# Ex 2:
# -----
# Define specific tile types and run a Mapper
t = [
    (40, TileType((0.5, 0.9, 0.3), 0.2, 1)),
    (10, TileType((0.15, 0.6, 0.2), [0.3, 0.7], [0, 1])),
    (20, TileType((0.35, 0.35, 0.35), [0.2, 0.4], [1, 2])),
    (30, TileType((0.0, 0.05, 0.75), [0.2, 0.45], [0, 3])),
]
MapGenerator.run(width=150, height=150, tile_types=t)

# Ex 3:
# -----
# Export result (with each process step)
MapGenerator.run(width=150, height=150, with_animation=True,
                 export_file='map-gen.gif')

# Ex 4:
# -----
# Set specific generation parameters
MapGenerator.run(width=150, height=150, smooth_size=[6, 4, 2],
                 earthifications=1, clean_up=2)
