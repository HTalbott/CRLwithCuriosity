#!/usr/bin/env python3
"""Trajectory plot across envs (ant_ball, humanoid, reacher, ant_u_maze).

Reads local sweep logs, produces a 2x2 grid with mean±1σ curves per variant.
Reacher panel shows a placeholder message since no runs are completed yet.

Usage: python plot_env_comparison.py [metric]
  metric: 'reward' (default) or 'success_any'
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/saga/reflearn/crl/JaxGCRL")
LOGS = ROOT / "sweep_logs"
OUT = ROOT / "results/report"

SMOOTH_WINDOW = 15
COLORS = {
    "Vanilla CRL":             "#1f77b4",
    "CRL+EMA":                 "#ff7f0e",
    "CRL+EMA+Goal (reg=1e-7)": "#2ca02c",
}

METRICS = {
    "reward": {
        "key": "eval/episode_reward",
        "ylabel": "Eval episode reward",
        "title": "Eval reward across training — env ablation (mean $\\pm$ 1$\\sigma$, 4 seeds)",
        "out_stem": "env_comparison_curves",
        "scale": 1.0,
    },
    "success_any": {
        "key": "eval/episode_success_any",
        "ylabel": "Success rate (%)",
        "title": "Episode success rate across training — env ablation (mean $\\pm$ 1$\\sigma$, 4 seeds)",
        "out_stem": "env_comparison_success_any_curves",
        "scale": 100.0,
    },
}

METRIC = sys.argv[1] if len(sys.argv) > 1 else "reward"
CFG = METRICS[METRIC]
EVAL_RE = re.compile(rf"step: (\d+),\s+{re.escape(CFG['key'])}: ([-0-9.]+)")

def load_log(path: Path) -> pd.DataFrame:
    steps, rewards = [], []
    with path.open() as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                rewards.append(float(m.group(2)) * CFG["scale"])
    return pd.DataFrame({"step": steps, "reward": rewards})

def variant_matrix(files):
    dfs = [load_log(f) for f in files]
    n = min(len(df) for df in dfs)
    step = dfs[0]["step"].iloc[:n].to_numpy(dtype=float)
    rewards = np.column_stack([df["reward"].iloc[:n].to_numpy(dtype=float) for df in dfs])
    return step, rewards

PANELS = [
    # (title, variants: {label: [log paths]})
    ("Ant Ball (70M)", {
        "Vanilla CRL": [LOGS / f"crl_ab70_s{s}.attempt1.log" for s in range(4)],
        "CRL+EMA":     [LOGS / f"crl_ema_ab70_s{s}.attempt1.log" for s in range(4)],
        "CRL+EMA+Goal (reg=1e-7)": [
            LOGS / "crl_ema_goal_anneal10m_gated_reg1e-7_s0.log",
            LOGS / "crl_ema_goal_anneal10m_gated_reg1e-7_s1.attempt1.log",
            LOGS / "crl_ema_goal_anneal10m_gated_reg1e-7_s2.attempt1.log",
            LOGS / "crl_ema_goal_anneal10m_gated_reg1e-7_s3.attempt1.log",
        ],
    }),
    ("Humanoid (50M)", {
        "Vanilla CRL": [LOGS / f"crl_hu50_s{s}.attempt1.log" for s in range(4)],
        "CRL+EMA": [
            LOGS / "crl_ema_hu50_s0.attempt2.log",  # attempt1 SIGSEGV at 33M
            LOGS / "crl_ema_hu50_s1.attempt1.log",
            LOGS / "crl_ema_hu50_s2.attempt1.log",
            LOGS / "crl_ema_hu50_s3.attempt1.log",
        ],
        "CRL+EMA+Goal (reg=1e-7)": [
            LOGS / "crl_ema_goal_hu50_reg1e-7_s0.attempt1.log",
            LOGS / "crl_ema_goal_hu50_reg1e-7_s1.attempt2.log",  # attempt1 SIGSEGV
            LOGS / "crl_ema_goal_hu50_reg1e-7_s2.attempt1.log",
            LOGS / "crl_ema_goal_hu50_reg1e-7_s3.attempt1.log",
        ],
    }),
    ("Reacher", None),  # placeholder
    ("Ant U-Maze (50M)", {
        "Vanilla CRL": [LOGS / f"crl_am50_s{s}.attempt1.log" for s in range(4)],
        "CRL+EMA":     [LOGS / f"crl_ema_am50_s{s}.attempt1.log" for s in range(4)],
        "CRL+EMA+Goal (reg=1e-7)": [
            LOGS / f"crl_ema_goal_am50_reg1e-7_s{s}.attempt1.log" for s in range(4)
        ],
    }),
]

def plot_panel(ax, title, variants):
    if variants is None:
        ax.text(0.5, 0.5, "No Reacher runs yet", ha="center", va="center",
                fontsize=14, color="gray", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        return
    for label, files in variants.items():
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"  [WARN] {title}: missing {len(missing)} file(s) for {label}: {missing[0]}")
            continue
        step, rewards = variant_matrix(files)
        smoothed = pd.DataFrame(rewards).rolling(
            window=SMOOTH_WINDOW, min_periods=1, center=True
        ).mean().to_numpy()
        mean = smoothed.mean(axis=1)
        std = smoothed.std(axis=1, ddof=1)
        x = step / 1e6
        color = COLORS[label]
        ax.plot(x, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel("Env steps (millions)")
    ax.set_ylabel(CFG["ylabel"])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()
    for ax, (title, variants) in zip(axes_flat, PANELS):
        plot_panel(ax, title, variants)

    # Single shared legend from first populated axis
    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   frameon=True, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(CFG["title"], y=1.05, fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"{CFG['out_stem']}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)

if __name__ == "__main__":
    main()
