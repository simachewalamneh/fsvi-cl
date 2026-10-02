import numpy as np
import jax, jax.numpy as jnp
from fsvi_cl.objectives import kl_diag_gauss, weight_kl
from fsvi_cl.models import init_params, forward, function_gaussian
from fsvi_cl.data import make_synthetic_tasks
from fsvi_cl.train import Config, run


def test_kl_zero_for_identical_gaussians():
    m, v = jnp.array([0.3, -1.0]), jnp.array([0.5, 2.0])
    assert abs(float(kl_diag_gauss(m, v, m, v))) < 1e-6


def test_kl_matches_closed_form_1d():
    # KL(N(0,1)||N(1,4)) = 0.5*(log 4 + (1+1)/4 - 1)
    got = float(kl_diag_gauss(jnp.array([0.0]), jnp.array([1.0]), jnp.array([1.0]), jnp.array([4.0])))
    assert abs(got - 0.5 * (np.log(4.0) + 0.5 - 1.0)) < 1e-6


def test_weight_kl_zero_against_self_and_positive_against_prior():
    p = init_params(jax.random.PRNGKey(0), [4, 3, 2])
    assert abs(float(weight_kl(p, p))) < 1e-4
    assert float(weight_kl(p, None)) > 0


def test_function_gaussian_shapes_and_positive_var():
    p = init_params(jax.random.PRNGKey(0), [4, 3, 4])
    m, v = function_gaussian(p, jnp.ones((5, 4)), jax.random.PRNGKey(1), 8, task=1)
    assert m.shape == v.shape == (5, 2) and bool(jnp.all(v > 0))


def test_all_methods_run_and_are_finite():
    tasks = make_synthetic_tasks(n_tasks=3, dim=16, n_train=256, n_test=128)
    for method in ("finetune", "vcl", "fsvi"):
        r = run(tasks, Config(method=method, hidden=(32,), epochs=2, batch=64), log=lambda *_: None)
        R = np.array(r["acc_matrix"])
        assert R.shape == (3, 3) and np.isfinite(R).all() and 0.0 <= R.min() and R.max() <= 1.0
