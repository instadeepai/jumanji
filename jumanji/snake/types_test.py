from jumanji.snake.types import Position


def test_position__eq() -> None:
    """Validates the equality check of the Position type.
    Checks whether some positions are equal or not.
    """
    pos1 = Position(3, 5)
    pos2 = Position(3, 5)
    assert pos1 == pos2
    pos3 = Position(4, 5)
    assert pos3 != pos1
