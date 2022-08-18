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

import shlex
from codecs import open
from subprocess import check_call
from typing import List

import setuptools
from setuptools import setup
from setuptools.command.develop import develop

__version__ = "0.0.0"


def read_requirements(*files: str) -> List[str]:
    """Returns content of given requirements file."""
    return [
        line
        for file in files
        for line in open(file)
        if not (line.startswith("#") or line.startswith("--"))
    ]


class PostDevelopCommand(develop):
    def run(self) -> None:
        try:
            check_call(shlex.split("pre-commit install"))
            check_call(shlex.split("pre-commit install --hook-type commit-msg"))
        except Exception as e:
            print("Unable to run 'pre-commit install'", e)
        develop.run(self)


setup(
    name="jumanji",
    version=__version__,
    description="Suite of Reinforcement Learning environments",
    author="InstaDeep",
    url="https://github.com/instadeepai/jumanji/",
    packages=setuptools.find_packages(),
    zip_safe=False,
    install_requires=read_requirements("./requirements/requirements.txt"),
    extras_require={
        "dev": read_requirements("./requirements/requirements-dev.txt"),
    },
    cmdclass={"develop": PostDevelopCommand},
    include_package_data=True,
)
