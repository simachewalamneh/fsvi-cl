"""Mean-field Gaussian Bayesian MLP with a multi-head output (2 logits per task)."""
from __future__ import annotations
import jax
import jax.numpy as jnp

Params = list  # list of {"w_mu","w_rho","b_mu","b_rho"} dicts, one per layer


def softplus_inv(x: float) -> float:
    return float(jnp.log(jnp.expm1(x)))


def init_params(key, sizes, init_sigma: float = 0.05) -> Params:
    """sizes = [in, h1, ..., out]. Means: He-normal. Std (via softplus(rho)): init_sigma."""
    rho0 = softplus_inv(init_sigma)
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        params.append({
            "w_mu": jax.random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din),
            "w_rho": jnp.full((din, dout), rho0),
            "b_mu": jnp.zeros((dout,)),
            "b_rho": jnp.full((dout,), rho0),
        })
    return params


def forward(params: Params, x, key=None):
    """One forward pass. key=None -> use the posterior mean; else sample one weight draw."""
    h = x
    last = len(params) - 1
    for i, layer in enumerate(params):
        if key is None:
            w, b = layer["w_mu"], layer["b_mu"]
        else:
            key, k1, k2 = jax.random.split(key, 3)
            w = layer["w_mu"] + jax.nn.softplus(layer["w_rho"]) * jax.random.normal(k1, layer["w_mu"].shape)
            b = layer["b_mu"] + jax.nn.softplus(layer["b_rho"]) * jax.random.normal(k2, layer["b_mu"].shape)
        h = h @ w + b
        if i < last:
            h = jax.nn.relu(h)
    return h


def head(logits, task: int):
    """Slice the 2-logit head of `task` from the multi-head output."""
    return logits[..., 2 * task: 2 * task + 2]


def sampled_outputs(params: Params, x, key, num_samples: int):
    """(S, N, K) outputs for S independent weight draws."""
    keys = jax.random.split(key, num_samples)
    return jax.vmap(lambda k: forward(params, x, k))(keys)


def function_gaussian(params: Params, x, key, num_samples: int, task: int, jitter: float = 1e-3):
    """Diagonal Gaussian over the task-head outputs at inputs x, estimated from weight samples."""
    outs = head(sampled_outputs(params, x, key, num_samples), task)
    return outs.mean(0), outs.var(0) + jitter
