import jax
import jax.numpy as jnp
def tree_to_float64(tree):
    # config.update("jax_enable_x64", True)
    return jax.tree.map(
        lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        tree
    )

def tree_to_float32(tree):
    # config.update("jax_enable_x64", True)
    return jax.tree.map(
        lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        tree
    )

def tree_subtraction(tree_1, tree_2):
    # config.update("jax_enable_x64", True)
    return jax.tree.map(
        lambda x, y: x - y,
        tree_1, tree_2
    )

def tree_max_diff(tree_1, tree_2):
    # config.update("jax_enable_x64", True)
    return jax.tree.map(
        lambda x, y: jnp.max(jnp.abs(x - y)),
        tree_1, tree_2
    )

def tree_allclose(tree1, tree2, tol=0.0):
    # compute per-leaf max difference
    diffs = jax.tree.map(lambda x, y: jnp.max(jnp.abs(x - y)), tree1, tree2)
    # flatten into a list of leaves
    leaves = jax.tree.leaves(diffs)
    # check if all differences are <= tol
    return all(float(d) <= tol for d in leaves)