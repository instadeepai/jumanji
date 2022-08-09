# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
