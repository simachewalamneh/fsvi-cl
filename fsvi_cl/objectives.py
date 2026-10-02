"""KL terms used by the regularisers."""
from __future__ import annotations
import jax
import jax.numpy as jnp


def kl_diag_gauss(mq, vq, mp, vp):
    """KL( N(mq, diag vq) || N(mp, diag vp) ), summed over all dimensions."""
    return 0.5 * jnp.sum(jnp.log(vp / vq) + (vq + (mq - mp) ** 2) / vp - 1.0)


def weight_kl(params, ref_params=None, prior_std: float = 1.0):
    """Parameter-space KL(q || ref). ref_params=None -> zero-mean N(0, prior_std^2) prior."""
    total = 0.0
    for i, layer in enumerate(params):
        for name in ("w", "b"):
            mu, sd = layer[f"{name}_mu"], jax.nn.softplus(layer[f"{name}_rho"])
            if ref_params is None:
                mp, sp = jnp.zeros_like(mu), jnp.full_like(sd, prior_std)
            else:
                r = ref_params[i]
                mp, sp = r[f"{name}_mu"], jax.nn.softplus(r[f"{name}_rho"])
            total += kl_diag_gauss(mu, sd ** 2, mp, sp ** 2)
    return total
