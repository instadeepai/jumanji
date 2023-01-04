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

import os
from typing import List

import setuptools
from setuptools import setup

__version__ = "0.1.4"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_requirements(path: str) -> List[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


setup(
    name="jumanji",
    version=__version__,
    author="InstaDeep",
    author_email="hello@instadeep.com",
    description="Industry-Driven Hardware-Accelerated RL Environments",
    license="Apache 2.0",
    url="https://github.com/instadeepai/jumanji/",
    long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
    long_description_content_type="text/markdown",
    keywords="reinforcement-learning python jax",
    packages=setuptools.find_packages(exclude=["*testing"]),
    python_requires=">=3.7",
    install_requires=_parse_requirements(
        os.path.join(_CURRENT_DIR, "requirements", "requirements.txt")
    ),
    extras_require={
        "dev": _parse_requirements(
            os.path.join(_CURRENT_DIR, "requirements", "requirements-dev.txt")
        ),
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
    zip_safe=False,
    include_package_data=True,
)
