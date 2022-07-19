import jax.numpy as jnp

from jumanji.jax.binpack.space import Space


def test_space__astype() -> None:
    space_int = Space(x1=0, x2=1, y1=0, y2=1, z1=0, z2=1)
    assert isinstance(space_int.x1, int)
    space_jnp_int = space_int.astype(jnp.int32)
    assert space_jnp_int.x1.dtype == jnp.int32
    space_jnp_float = space_int.astype(jnp.float32)
    assert space_jnp_float.x1.dtype == jnp.float32
