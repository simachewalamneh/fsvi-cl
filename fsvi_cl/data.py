"""Task streams. Every task is a binary problem with local labels {0, 1}."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Task:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    name: str = ""


def make_synthetic_tasks(n_tasks: int = 5, dim: int = 64, n_train: int = 600, n_test: int = 300,
                         noise: float = 1.0, separation: float = 2.0, seed: int = 0) -> list[Task]:
    """Gaussian-cluster tasks sharing one input space (so they can interfere). For tests and quick checks."""
    rng = np.random.default_rng(seed)
    tasks = []
    for t in range(n_tasks):
        protos = rng.normal(size=(2, dim))
        protos = separation * protos / np.linalg.norm(protos, axis=1, keepdims=True) * np.sqrt(dim) / 2
        def draw(n):
            y = rng.integers(0, 2, n)
            x = protos[y] + noise * rng.normal(size=(n, dim))
            return x.astype(np.float32), y.astype(np.int32)
        trx, try_ = draw(n_train)
        tex, tey = draw(n_test)
        tasks.append(Task(trx, try_, tex, tey, name=f"synthetic_{t}"))
    return tasks


def make_split_mnist(n_tasks: int = 5) -> list[Task]:
    """Split-MNIST: task t = digits (2t, 2t+1). Needs `keras` and internet on first call."""
    import os
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from keras.datasets import mnist
    (trx, try_), (tex, tey) = mnist.load_data()
    trx = trx.reshape(len(trx), -1).astype(np.float32) / 255.0
    tex = tex.reshape(len(tex), -1).astype(np.float32) / 255.0
    tasks = []
    for t in range(n_tasks):
        a, b = 2 * t, 2 * t + 1
        def pick(x, y):
            m = (y == a) | (y == b)
            return x[m], (y[m] == b).astype(np.int32)
        ax, ay = pick(trx, try_)
        bx, by = pick(tex, tey)
        tasks.append(Task(ax, ay, bx, by, name=f"mnist_{a}_{b}"))
    return tasks


def load_tasks(dataset: str, seed: int = 0, n_tasks: int = 5) -> list[Task]:
    if dataset == "synthetic":
        return make_synthetic_tasks(n_tasks=n_tasks, seed=seed)
    if dataset == "split_mnist":
        return make_split_mnist(n_tasks=n_tasks)
    raise ValueError(f"unknown dataset: {dataset}")
