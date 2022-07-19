from typing import TYPE_CHECKING, Any

import chex
import jax.numpy as jnp

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class Space:
    x1: chex.Numeric
    x2: chex.Numeric
    y1: chex.Numeric
    y2: chex.Numeric
    z1: chex.Numeric
    z2: chex.Numeric

    def astype(self, dtype: Any) -> "Space":
        space_dict = {
            key: jnp.asarray(value, dtype) for key, value in self.__dict__.items()
        }
        return Space(**space_dict)

    def __repr__(self) -> str:
        return (
            "Space(\n"
            f"\tx1={repr(self.x1)}, x2={repr(self.x2)},\n"
            f"\ty1={repr(self.y1)}, y2={repr(self.y2)},\n"
            f"\tz1={repr(self.z1)}, z2={repr(self.z2)},\n"
            ")"
        )
