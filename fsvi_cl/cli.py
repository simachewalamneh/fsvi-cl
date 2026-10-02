"""python -m fsvi_cl.cli --method fsvi --dataset split_mnist"""
from __future__ import annotations
import argparse, json, os
from .data import load_tasks
from .train import Config, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["finetune", "vcl", "fsvi"], default="fsvi")
    p.add_argument("--dataset", choices=["synthetic", "split_mnist"], default="synthetic")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--beta-first", type=float, default=0.01)
    p.add_argument("--context-size", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    cfg = Config(method=a.method, epochs=a.epochs, beta=a.beta, beta_first=a.beta_first,
                 context_size=a.context_size, seed=a.seed)
    res = run(load_tasks(a.dataset, seed=a.seed), cfg)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.dataset}_{a.method}_seed{a.seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"avg_final_acc={res['avg_final_acc']:.4f} avg_forgetting={res['avg_forgetting']:.4f} -> {path}")


if __name__ == "__main__":
    main()
