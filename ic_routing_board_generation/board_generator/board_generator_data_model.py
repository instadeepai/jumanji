from dataclasses import dataclass


@dataclass
class Position:
    # Class of 2D tuple of ints, indicating the size of an array or a 2D position or vector.
    x: int
    y: int
