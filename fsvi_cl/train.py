"""Sequential training loop for the three methods compared in stage 1.

methods
  finetune : deterministic fine-tuning on the posterior mean, no regulariser (forgetting lower bound)
  vcl      : parameter-space VI, KL(q_t || q_{t-1}) over the weights (VCL-style, no coreset)
  fsvi     : S-FSVI, KL over *function* outputs at stored context points of earlier tasks
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import jax
import jax.numpy as jnp
import optax

from .models import init_params, forward, head, sampled_outputs, function_gaussian
from .objectives import kl_diag_gauss, weight_kl


@dataclass
class Config:
    method: str = "fsvi"            # finetune | vcl | fsvi
    hidden: tuple = (100, 100)
    epochs: int = 5
    batch: int = 128
    lr: float = 1e-3
    mc_train: int = 4               # weight samples per training step
    mc_ctx: int = 16                # weight samples to estimate function-space Gaussians
    beta: float = 1.0               # weight of the continual-learning KL (tasks >= 1)
    beta_first: float = 0.01        # weight of the weight-space KL to the N(0, prior_std^2) prior (task 0)
    prior_std: float = 1.0
    init_sigma: float = 0.05
    context_size: int = 40          # context points kept per finished task (fsvi)
    seed: int = 0


def _make_step(cfg: Config, t: int, n_train: int, opt):
    def loss_fn(params, xb, yb, key, ref, ctxs, priors):
        if cfg.method == "finetune":
            logp = jax.nn.log_softmax(head(forward(params, xb), t))
            return -jnp.mean(jnp.sum(logp * jax.nn.one_hot(yb, 2), -1))
        k1, k2 = jax.random.split(key)
        outs = head(sampled_outputs(params, xb, k1, cfg.mc_train), t)       # (S, B, 2)
        logp = jax.nn.log_softmax(outs)
        ce = -jnp.mean(jnp.sum(logp * jax.nn.one_hot(yb, 2)[None], -1))
        if t == 0:
            reg, beta = weight_kl(params, None, cfg.prior_std), cfg.beta_first
        elif cfg.method == "vcl":
            reg, beta = weight_kl(params, ref, cfg.prior_std), cfg.beta
        else:  # fsvi, t >= 1: match earlier-task function distributions at their context points
            keys = jax.random.split(k2, t)
            reg = 0.0
            for s in range(t):
                m, v = function_gaussian(params, ctxs[s], keys[s], cfg.mc_ctx, s)
                reg += kl_diag_gauss(m, v, priors[s][0], priors[s][1])
            beta = cfg.beta
        return ce + beta * reg / n_train

    @jax.jit
    def step(params, opt_state, xb, yb, key, ref, ctxs, priors):
        loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb, key, ref, ctxs, priors)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
    return step


def evaluate(params, tasks) -> np.ndarray:
    accs = []
    for j, task in enumerate(tasks):
        pred = jnp.argmax(head(forward(params, jnp.asarray(task.test_x)), j), -1)
        accs.append(float(jnp.mean(pred == jnp.asarray(task.test_y))))
    return np.array(accs)


def run(tasks, cfg: Config, log=print) -> dict:
    T = len(tasks)
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)
    key, k0 = jax.random.split(key)
    params = init_params(k0, [tasks[0].train_x.shape[1], *cfg.hidden, 2 * T], cfg.init_sigma)
    contexts, R, losses = [], np.zeros((T, T)), []
    for t, task in enumerate(tasks):
        ref = params                                   # posterior after task t-1 (frozen snapshot)
        priors = []
        if cfg.method == "fsvi" and t > 0:             # previous posterior, evaluated at ALL earlier contexts
            for s, cx in enumerate(contexts):
                key, k = jax.random.split(key)
                priors.append(function_gaussian(ref, cx, k, cfg.mc_ctx, s))
        opt = optax.adam(cfg.lr)
        opt_state = opt.init(params)
        step = _make_step(cfg, t, len(task.train_x), opt)
        n = len(task.train_x)
        for ep in range(cfg.epochs):
            perm = rng.permutation(n)
            ep_loss = []
            for i in range(0, n - cfg.batch + 1, cfg.batch):
                idx = perm[i:i + cfg.batch]
                key, k = jax.random.split(key)
                params, opt_state, loss = step(params, opt_state, jnp.asarray(task.train_x[idx]),
                                               jnp.asarray(task.train_y[idx]), k, ref, contexts, priors)
                ep_loss.append(float(loss))
            losses.append((t, ep, float(np.mean(ep_loss))))
        sel = rng.choice(n, min(cfg.context_size, n), replace=False)
        contexts.append(jnp.asarray(task.train_x[sel]))
        R[t] = evaluate(params, tasks)
        log(f"[{cfg.method}] after task {t}: acc on tasks 0..{t} = {np.round(R[t, :t + 1], 3).tolist()}")
    forget = float(np.mean([R[j:T - 1, j].max() - R[T - 1, j] for j in range(T - 1)])) if T > 1 else 0.0
    return {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
            "acc_matrix": R.tolist(), "avg_final_acc": float(R[T - 1].mean()), "avg_forgetting": forget,
            "epoch_losses": losses}
