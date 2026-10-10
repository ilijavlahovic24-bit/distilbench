"""Generate report figures from saved results.

Reads:
  results/data_efficiency/*.json  -> data_efficiency.png
  results/exp1_compare_methods.json -> exp1_methods.png
  results/exp3_student_size.json  -> exp3_student_size.png
Writes PNGs to report/figures/.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("plot_results")

FIG_DIR = Path("report/figures")


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_data_efficiency(sweep_root: Path, out: Path) -> bool:
    families: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)

    for p in sorted(Path(sweep_root).glob("*.json")):
        data = _load_json(p)
        if not data or "accuracy" not in data:
            continue
        m = re.match(r"(.+?)_n(\d+)$", p.stem)
        if not m:
            continue
        family = m.group(1)
        n = int(m.group(2))
        families[family].append((n, data["accuracy"]))

    if not families:
        LOG.warning("No data-efficiency results found in %s", sweep_root)
        return False

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for family, points in sorted(families.items()):
        points = sorted(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=family)
    ax.set_xlabel("Number of training examples")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Data efficiency: CoT vs answer-only")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Wrote %s", out)
    return True


def plot_exp1(results_path: Path, out: Path) -> bool:
    table = _load_json(results_path)
    if not table:
        LOG.warning("No exp1 results at %s", results_path)
        return False

    methods = [m for m in table if "accuracy" in table[m]]
    if not methods:
        return False
    accs = [table[m]["accuracy"] for m in methods]

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(methods, accs, color=colors[:len(methods)])
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Exp1: method comparison (n={table[methods[0]]['n_train']})")
    ax.set_ylim(0, max(accs) * 1.2 + 0.02)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Wrote %s", out)
    return True


def plot_exp3(results_path: Path, out: Path) -> bool:
    table = _load_json(results_path)
    if not table:
        LOG.warning("No exp3 results at %s", results_path)
        return False

    keys = [k for k in table if "accuracy" in table[k]]
    if not keys:
        return False
    accs = [table[k]["accuracy"] for k in keys]

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(keys, accs, color="#4C72B0")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Exp3: student size ablation")
    ax.set_ylim(0, max(accs) * 1.2 + 0.02)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    LOG.info("Wrote %s", out)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path,
                   default=Path("results/data_efficiency"))
    p.add_argument("--exp1", type=Path,
                   default=Path("results/exp1_compare_methods.json"))
    p.add_argument("--exp3", type=Path,
                   default=Path("results/exp3_student_size.json"))
    p.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_data_efficiency(args.sweep_root, args.out_dir / "data_efficiency.png")
    plot_exp1(args.exp1, args.out_dir / "exp1_methods.png")
    plot_exp3(args.exp3, args.out_dir / "exp3_student_size.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())