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

import copy
from dataclasses import _is_dataclass_instance, asdict
from typing import Any, Union

from jumanji import specs
from jumanji.environments.routing.macvrp.types import (
    Node,
    Observation,
    PenalityCoeff,
    TimeWindow,
    Vehicle,
)


class BaseSpec(specs.Spec[Any]):
    def __init__(
        self,
        **kwargs: Any,
    ):
        self._specs = kwargs
        self.obj_type: Any = lambda args: None

    def __repr__(self) -> str:
        return_str = f"{self.obj_type}(\n"
        for spec_name, spec in self._specs.items():
            return_str += f". {spec_name}={repr(spec)},\n"
        return return_str

    def generate_value(self) -> Any:
        """Generate a value which conforms to this spec."""

        args = {}
        for spec_name, spec in self._specs.items():
            attr_name = spec_name.rsplit("_spec")[0]
            if spec is not None:
                args[attr_name] = spec.generate_value()
            else:
                args[attr_name] = None

        return self.obj_type(**args)

    def validate(self, value: Any) -> Any:
        """Checks if a this object conforms to the spec. In particular,
        it returns the Node if it conforms to the `Spec` and
        raises an error otherwise.
        Args:
            value: an instantiation of the object.
        Returns:
            an new instantiation of the object.
        Raises:
            ValueError: if value doesn't conform to this spec.
        """

        if type(value) is not dict:

            if _is_dataclass_instance(value):
                value = asdict(value)
            else:
                value = value._asdict()

        args = {}
        for spec_name, spec in self._specs.items():
            attr_name = spec_name.rsplit("_spec")[0]
            if spec is not None:
                args[attr_name] = spec.validate(value[attr_name])
            else:
                args[attr_name] = None

        return self.obj_type(**args)

    def replace(self, **kwargs: Any) -> Any:
        """Returns a new copy of `Spec` with specified attributes replaced.
        Args:
            **kwargs: Optional attributes to replace.
        Returns:
            A new copy of `Spec`.
        """

        all_kwargs = copy.copy(self._specs)
        all_kwargs.update(kwargs)

        return type(self)(**all_kwargs)


class NodeSpec(BaseSpec):
    def __init__(
        self,
        coordinates_spec: specs.BoundedArray,
        demands_spec: specs.BoundedArray,
    ):
        super().__init__(name="node")
        self._specs = {
            "coordinates_spec": coordinates_spec,
            "demands_spec": demands_spec,
        }
        self.obj_type = Node


class TimeWindowSpec(BaseSpec):
    def __init__(
        self,
        start_spec: specs.BoundedArray,
        end_spec: specs.BoundedArray,
    ):
        super().__init__(name="time_window")
        self._specs = {"start_spec": start_spec, "end_spec": end_spec}
        self.obj_type = TimeWindow


class PenalityCoeffSpec(BaseSpec):
    def __init__(
        self,
        early_spec: specs.BoundedArray,
        late_spec: specs.BoundedArray,
    ):
        super().__init__(name="penality_coeff")
        self._specs = {"early_spec": early_spec, "late_spec": late_spec}
        self.obj_type = PenalityCoeff


class VehicleSpec(BaseSpec):
    def __init__(
        self,
        positions_spec: specs.BoundedArray,
        local_times_spec: specs.BoundedArray,
        capacities_spec: specs.BoundedArray,
        distances_spec: Union[specs.BoundedArray, None] = None,
        time_penalties_spec: Union[specs.BoundedArray, None] = None,
    ):
        super().__init__(name="vehicle")
        self._specs = {
            "local_times_spec": local_times_spec,
            "positions_spec": positions_spec,
            "capacities_spec": capacities_spec,
            "distances_spec": distances_spec,
            "time_penalties_spec": time_penalties_spec,
        }
        self.obj_type = Vehicle


class ObservationSpec(BaseSpec):
    def __init__(
        self,
        nodes_spec: NodeSpec,
        windows_spec: TimeWindowSpec,
        coeffs_spec: PenalityCoeffSpec,
        other_vehicles_spec: VehicleSpec,
        main_vehicles_spec: VehicleSpec,
        action_mask_spec: specs.BoundedArray,
    ):
        super().__init__(name="observation")
        self._specs = {
            "nodes_spec": nodes_spec,
            "windows_spec": windows_spec,
            "coeffs_spec": coeffs_spec,
            "other_vehicles_spec": other_vehicles_spec,
            "main_vehicles_spec": main_vehicles_spec,
            "action_mask_spec": action_mask_spec,
        }
        self.obj_type = Observation
