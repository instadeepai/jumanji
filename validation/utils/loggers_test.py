from pytest import CaptureFixture

from validation.utils.loggers import TerminalLogger


def test_terminal_logger(capsys: CaptureFixture) -> None:
    """Check logging a dictionary and the corresponding string printed to terminal."""
    logger = TerminalLogger("terminal")
    values = {"gradient_norm": 32.4452, "total_return": 14.0}
    logger.write(values)
    assert (
        capsys.readouterr().out
        == "[Terminal] Gradient Norm = 32.445 | Total Return = 14.000\n"
    )
