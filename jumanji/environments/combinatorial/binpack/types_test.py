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

import pytest

from jumanji.environments.combinatorial.binpack.space import Space
from jumanji.environments.combinatorial.binpack.types import (
    Item,
    Location,
    item_fits_in_item,
    item_from_space,
    item_volume,
    location_from_space,
    space_from_item_and_location,
)


def test__item_from_space(space: Space) -> None:
    item = item_from_space(space)
    assert isinstance(item, Item)
    assert item.x_len == item.y_len == item.z_len == 1


def test__location_from_space(space: Space) -> None:
    location = location_from_space(space)
    assert isinstance(location, Location)
    assert location.x == location.y == location.z == 0


def test__item_fits_in_item() -> None:
    item_1 = Item(1, 0, 1)
    item_2 = Item(1, 2, 2)
    assert item_fits_in_item(item_1, item_1)
    assert item_fits_in_item(item_1, item_2)
    assert not item_fits_in_item(item_2, item_1)


def test__item_volume() -> None:
    item_1 = Item(1, 0, 1)
    item_2 = Item(1, 2, 2)
    assert item_volume(item_1) == 0
    assert item_volume(item_2) == 4


@pytest.mark.parametrize(
    ["item", "location", "expected_space"],
    [
        (Item(2, 3, 4), Location(1, 4, 5), Space(x1=1, x2=3, y1=4, y2=7, z1=5, z2=9)),
        (
            Item(4, 1, 6),
            Location(10, 5, 3),
            Space(x1=10, x2=14, y1=5, y2=6, z1=3, z2=9),
        ),
    ],
)
def test__space_from_item_and_location(
    item: Item, location: Location, expected_space: Space
) -> None:
    space = space_from_item_and_location(item, location)
    assert space == expected_space
