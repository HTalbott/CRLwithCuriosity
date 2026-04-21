#!/usr/bin/env python3
"""Fetch wandb run histories and produce a report-ready summary table
and a mean ± 1σ eval-reward curve plot.

Configured for the ant_ball 70M sweep (vanilla / EMA / EMA+Goal reg=1e-7).
Swap SPEC to re-run on the humanoid 50M sweep once it finishes.

Outputs (next to this script):
  <prefix>_raw.csv       -- all fetched eval rows, one per (variant, seed, step)
  <prefix>_table.csv     -- per-seed final reward + mean/std across seeds
  <prefix>_table.md      -- same table in markdown
  <prefix>_curves.png    -- mean ± 1σ curves, one line per variant
  <prefix>_curves.pdf    -- same, vector
"""
from __future__ import annotations
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

ENTITY = "lyrneos-universty-of-michigan"
PROJECT = "jaxgcrl"
OUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUT_DIR / "_cache"
LAST_N_EVALS = 20
HISTORY_SAMPLES = 500  # >= actual eval count per run
MAX_WORKERS = 6
SMOOTH_WINDOW = 15  # rolling-mean window in eval steps (each eval ≈ 350k env steps)

# Pick one by editing ACTIVE, or pass the key as the first CLI arg.
ACTIVE = "ant_ball_70m"

SPECS = {
    "ant_ball_70m": {
        "title": "Ant_ball 70M: eval reward across training",
        "prefix": "ant_ball_70m",
        "variants": [
            ("Vanilla CRL",
             "ant_ball_70m",
             [f"crl_ab70_s{s}" for s in range(4)],
             "#1f77b4"),
            ("CRL+EMA",
             "ant_ball_70m_anneal10m",
             [f"crl_ema_ab70_s{s}" for s in range(4)],
             "#ff7f0e"),
            ("CRL+EMA+Goal (reg=1e-7)",
             "ant_ball_70m_anneal10m",
             [f"crl_ema_goal_anneal10m_gated_reg1e-7_s{s}" for s in range(4)],
             "#2ca02c"),
        ],
    },
    "humanoid_50m": {
        "title": "Humanoid 50M: eval reward across training",
        "prefix": "humanoid_50m",
        "variants": [
            ("Vanilla CRL",
             "humanoid_50m_anneal10m",
             [f"crl_hu50_s{s}" for s in range(4)],
             "#1f77b4"),
            ("CRL+EMA",
             "humanoid_50m_anneal10m",
             [f"crl_ema_hu50_s{s}" for s in range(4)],
             "#ff7f0e"),
            ("CRL+EMA+Goal (reg=1e-7)",
             "humanoid_50m_anneal10m",
             [f"crl_ema_goal_hu50_reg1e-7_s{s}" for s in range(4)],
             "#2ca02c"),
        ],
    },
}

SPEC = SPECS[sys.argv[1] if len(sys.argv) > 1 else ACTIVE]


def find_finished_run(api, group, name):
    runs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"group": group, "displayName": name, "state": "finished"},
    ))
    if not runs:
        raise RuntimeError(f"No finished run for group={group!r} name={name!r}")
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs[0]


def pull_eval_df(run):
    """Fast fetch of eval rows via the aggregate history endpoint. Cached by run.id."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{run.id}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    # eval/episode_success is a duplicate of eval/episode_reward in ant_ball;
    # eval/episode_success_any is the true binary success rate (0-1).
    df = run.history(
        keys=["step", "eval/episode_reward", "eval/episode_success_any"],
        samples=HISTORY_SAMPLES,
        pandas=True,
    )
    df = df.dropna(subset=["eval/episode_reward"]).reset_index(drop=True)
    df["eval_idx"] = df.index
    df["step"] = df["step"].astype(float)
    df["eval/episode_reward"] = df["eval/episode_reward"].astype(float)
    if "eval/episode_success_any" in df.columns:
        df["eval/episode_success_any"] = df["eval/episode_success_any"].astype(float)
    cache.write_bytes(pickle.dumps(df))
    return df


def build_variant_matrix(dfs):
    """Stack seed reward curves into [n_evals, n_seeds]; step axis from seed 0.
    Length-matches to the shortest seed in case a run ended early."""
    n = min(len(df) for df in dfs)
    step = dfs[0]["step"].iloc[:n].to_numpy(dtype=float)
    rewards = np.column_stack([df["eval/episode_reward"].iloc[:n].to_numpy(dtype=float) for df in dfs])
    return step, rewards


def main():
    api = wandb.Api(timeout=60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []  # (label, seed_idx, name, group)
    for label, group, names, _color in SPEC["variants"]:
        for seed_idx, name in enumerate(names):
            jobs.append((label, seed_idx, name, group))

    print(f"Resolving {len(jobs)} run IDs...", flush=True)
    runs = {}
    for label, seed_idx, name, group in jobs:
        runs[(label, seed_idx)] = find_finished_run(api, group, name)
        print(f"  resolved {label:<26s} seed={seed_idx}  {runs[(label, seed_idx)].name}", flush=True)

    print(f"\nFetching history (parallel, workers={MAX_WORKERS})...", flush=True)
    def _fetch(key):
        label, seed_idx = key
        run = runs[key]
        df = pull_eval_df(run)
        df["variant"] = label
        df["seed"] = seed_idx
        df["run_id"] = run.id
        df["run_name"] = run.name
        return key, df

    dfs_by_key = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for key, df in ex.map(_fetch, list(runs.keys())):
            dfs_by_key[key] = df
            run = runs[key]
            print(f"  pulled  {key[0]:<26s} seed={key[1]}  {run.name:<55s} "
                  f"evals={len(df):>3d}  final={df['eval/episode_reward'].iloc[-1]:.2f}", flush=True)

    collected = {}
    for label, _g, names, _c in SPEC["variants"]:
        collected[label] = [dfs_by_key[(label, i)] for i in range(len(names))]

    # Raw dump for reproducibility
    raw = pd.concat([df for s in collected.values() for df in s], ignore_index=True)
    raw_path = OUT_DIR / f"{SPEC['prefix']}_raw.csv"
    raw.to_csv(raw_path, index=False)
    print(f"wrote {raw_path}")

    # Summary table: last-N eval mean per seed, then mean/std across seeds
    rows = []
    for label, _g, _n, _c in SPEC["variants"]:
        per_seed = np.array([
            df["eval/episode_reward"].tail(LAST_N_EVALS).mean()
            for df in collected[label]
        ])
        row = {"Variant": label}
        for i, v in enumerate(per_seed):
            row[f"seed {i}"] = round(float(v), 2)
        row["mean"] = round(float(per_seed.mean()), 2)
        row["std"] = round(float(per_seed.std(ddof=1)), 2)
        rows.append(row)
    tbl = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"{SPEC['prefix']}_table.csv"
    md_path = OUT_DIR / f"{SPEC['prefix']}_table.md"
    tbl.to_csv(csv_path, index=False)
    with md_path.open("w") as f:
        f.write(f"# {SPEC['title']}\n\n")
        f.write(f"Reward = mean over the last {LAST_N_EVALS} eval checkpoints per seed. "
                f"`mean` / `std` are across the 4 seeds (sample std, ddof=1).\n\n")
        f.write(tbl.to_markdown(index=False))
        f.write("\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print()
    print(tbl.to_string(index=False))
    print()

    # Curve plot: smooth each seed first, then mean ± 1σ across seeds
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, _g, _n, color in SPEC["variants"]:
        step, rewards = build_variant_matrix(collected[label])
        smoothed = pd.DataFrame(rewards).rolling(
            window=SMOOTH_WINDOW, min_periods=1, center=True
        ).mean().to_numpy()
        mean = smoothed.mean(axis=1)
        std = smoothed.std(axis=1, ddof=1)
        x = step / 1e6
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel("Env steps (millions)")
    ax.set_ylabel(f"Eval episode reward (rolling mean, window={SMOOTH_WINDOW} evals)")
    ax.set_title(SPEC["title"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{SPEC['prefix']}_curves.{ext}"
        fig.savefig(p, dpi=150)
        print(f"wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
