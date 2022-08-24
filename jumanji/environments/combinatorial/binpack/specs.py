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

from typing import Any

from jumanji import specs
from jumanji.environments.combinatorial.binpack.types import EMS, Item, Observation


class EMSSpec(specs.Spec[EMS]):
    def __init__(
        self,
        x1_spec: specs.BoundedArray,
        x2_spec: specs.BoundedArray,
        y1_spec: specs.BoundedArray,
        y2_spec: specs.BoundedArray,
        z1_spec: specs.BoundedArray,
        z2_spec: specs.BoundedArray,
    ):
        """Instantiate the specifications of the ems.

        Args:
            x1_spec: BoundedArray spec defining the specifications of the x1 coordinate of all ems.
            x2_spec: BoundedArray spec defining the specifications of the x2 coordinate of all ems.
            y1_spec: BoundedArray spec defining the specifications of the y1 coordinate of all ems.
            y2_spec: BoundedArray spec defining the specifications of the y2 coordinate of all ems.
            z1_spec: BoundedArray spec defining the specifications of the z1 coordinate of all ems.
            z2_spec: BoundedArray spec defining the specifications of the z2 coordinate of all ems.
        """
        super().__init__(name="ems")
        self.x1_spec = x1_spec
        self.x2_spec = x2_spec
        self.y1_spec = y1_spec
        self.y2_spec = y2_spec
        self.z1_spec = z1_spec
        self.z2_spec = z2_spec

    def __repr__(self) -> str:
        return (
            "EMSSpec(\n"
            f"\tx1_spec={repr(self.x1_spec)},\n"
            f"\tx2_spec={repr(self.x2_spec)},\n"
            f"\ty1_spec={repr(self.y1_spec)},\n"
            f"\ty2_spec={repr(self.y2_spec)},\n"
            f"\tz1_spec={repr(self.z1_spec)},\n"
            f"\tz2_spec={repr(self.z2_spec)},\n"
            ")"
        )

    def generate_value(self) -> EMS:
        """Generate a value which conforms to this spec."""
        return EMS(
            x1=self.x1_spec.generate_value(),
            x2=self.x2_spec.generate_value(),
            y1=self.y1_spec.generate_value(),
            y2=self.y2_spec.generate_value(),
            z1=self.z1_spec.generate_value(),
            z2=self.z2_spec.generate_value(),
        )

    def validate(self, value: EMS) -> EMS:
        """Checks if an EMS conforms to the EMS spec.

        Args:
            value: an EMS.

        Returns:
            value if no error is raised.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        value = EMS(
            x1=self.x1_spec.validate(value.x1),
            x2=self.x2_spec.validate(value.x2),
            y1=self.y1_spec.validate(value.y1),
            y2=self.y2_spec.validate(value.y2),
            z1=self.z1_spec.validate(value.z1),
            z2=self.z2_spec.validate(value.z2),
        )
        return value

    def replace(self, **kwargs: Any) -> "EMSSpec":
        """Returns a new copy of `EMSSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `EMSSpec`.
        """
        all_kwargs = {
            "x1_spec": self.x1_spec,
            "x2_spec": self.x2_spec,
            "y1_spec": self.y1_spec,
            "y2_spec": self.y2_spec,
            "z1_spec": self.z1_spec,
            "z2_spec": self.z2_spec,
        }
        all_kwargs.update(kwargs)
        return EMSSpec(**all_kwargs)  # type: ignore


class ItemSpec(specs.Spec[Item]):
    def __init__(
        self,
        x_len_spec: specs.BoundedArray,
        y_len_spec: specs.BoundedArray,
        z_len_spec: specs.BoundedArray,
    ):
        """Instantiate the specifications of items.

        Args:
            x_len_spec: BoundedArray spec defining what the length of an item on the x-axis can be.
            y_len_spec: BoundedArray spec defining what the length of an item on the y-axis can be.
            z_len_spec: BoundedArray spec defining what the length of an item on the z-axis can be.
        """
        super().__init__(name="items")
        self.x_len_spec = x_len_spec
        self.y_len_spec = y_len_spec
        self.z_len_spec = z_len_spec

    def __repr__(self) -> str:
        return (
            "ItemSpec(\n"
            f"\tx_len_spec={repr(self.x_len_spec)},\n"
            f"\ty_len_spec={repr(self.y_len_spec)},\n"
            f"\tz_len_spec={repr(self.z_len_spec)},\n"
            ")"
        )

    def generate_value(self) -> Item:
        """Generate a value which conforms to this spec."""
        return Item(
            x_len=self.x_len_spec.generate_value(),
            y_len=self.y_len_spec.generate_value(),
            z_len=self.z_len_spec.generate_value(),
        )

    def validate(self, value: Item) -> Item:
        """Checks if an Item conforms to the Item spec.

        Args:
            value: an Item.

        Returns:
            value if no error is raised.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        value = Item(
            x_len=self.x_len_spec.validate(value.x_len),
            y_len=self.y_len_spec.validate(value.y_len),
            z_len=self.z_len_spec.validate(value.z_len),
        )
        return value

    def replace(self, **kwargs: Any) -> "ItemSpec":
        """Returns a new copy of `ItemSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ItemSpec`.
        """
        all_kwargs = {
            "x_len_spec": self.x_len_spec,
            "y_len_spec": self.y_len_spec,
            "z_len_spec": self.z_len_spec,
        }
        all_kwargs.update(kwargs)
        return ItemSpec(**all_kwargs)  # type: ignore


class ObservationSpec(specs.Spec[Observation]):
    def __init__(
        self,
        ems_spec: EMSSpec,
        ems_mask_spec: specs.BoundedArray,
        items_spec: ItemSpec,
        items_mask_spec: specs.BoundedArray,
        items_placed_spec: specs.BoundedArray,
        action_mask_spec: specs.BoundedArray,
    ):
        """Instantiate the specifications of a BinPack observation.

        Args:
            ems_spec: EMSSpec defining the specifications of the ems.
            ems_mask_spec: BoundedArray spec defining the specifications of the ems mask.
            items_spec: ItemSpec defining the specifications of the items.
            items_mask_spec: BoundedArray spec defining the specifications of the item_mask.
            items_placed_spec: BoundedArray spec defining the specifications of the item_placed
                mask.
            action_mask_spec: BoundedArray spec defining the specifications of the action mask.
        """
        super().__init__(name="observation")
        self.ems_spec = ems_spec
        self.ems_mask_spec = ems_mask_spec
        self.items_spec = items_spec
        self.items_mask_spec = items_mask_spec
        self.items_placed_spec = items_placed_spec
        self.action_mask_spec = action_mask_spec

    def __repr__(self) -> str:
        return (
            "ObservationSpec(\n"
            f"\tems_spec={repr(self.ems_spec)},\n"
            f"\tems_mask_spec={repr(self.ems_mask_spec)},\n"
            f"\titems_spec={repr(self.items_spec)},\n"
            f"\titems_mask_spec={repr(self.items_mask_spec)},\n"
            f"\titems_placed_spec={repr(self.items_placed_spec)},\n"
            f"\taction_mask_spec={repr(self.action_mask_spec)},\n"
            ")"
        )

    def generate_value(self) -> Observation:
        """Generate a value which conforms to this spec."""
        return Observation(
            ems=self.ems_spec.generate_value(),
            ems_mask=self.ems_mask_spec.generate_value(),
            items=self.items_spec.generate_value(),
            items_mask=self.items_mask_spec.generate_value(),
            items_placed=self.items_placed_spec.generate_value(),
            action_mask=self.action_mask_spec.generate_value(),
        )

    def validate(self, value: Observation) -> Observation:
        """Checks if a BinPack Observation conforms to the spec.

        Args:
            value: a BinPack Observation.

        Returns:
            value if no error is raised.

        Raises:
            ValueError: if value doesn't conform to this spec.
        """
        observation = Observation(
            ems=self.ems_spec.validate(value.ems),
            ems_mask=self.ems_mask_spec.validate(value.ems_mask),
            items=self.items_spec.validate(value.items),
            items_mask=self.items_mask_spec.validate(value.items_mask),
            items_placed=self.items_placed_spec.validate(value.items_placed),
            action_mask=self.action_mask_spec.validate(value.action_mask),
        )
        return observation

    def replace(self, **kwargs: Any) -> "ObservationSpec":
        """Returns a new copy of `ObservationSpec` with specified attributes replaced.

        Args:
            **kwargs: Optional attributes to replace.

        Returns:
            A new copy of `ObservationSpec`.
        """
        all_kwargs = {
            "ems_spec": self.ems_spec,
            "ems_mask_spec": self.ems_mask_spec,
            "items_spec": self.items_spec,
            "items_mask_spec": self.items_mask_spec,
            "items_placed_spec": self.items_placed_spec,
            "action_mask_spec": self.action_mask_spec,
        }
        all_kwargs.update(kwargs)
        return ObservationSpec(**all_kwargs)  # type: ignore
