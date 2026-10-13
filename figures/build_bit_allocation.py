"""Precision allocation figure, from exploratory measured channels and costs."""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_theme import setup,finish,NAVY,OCHRE,TEAL
setup()

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("QRU_DATA_DIR", ROOT / "reference_data")) / "channel_study"


def run():
    rows = list(csv.DictReader((DATA / "precision.csv").open()))
    costs = list(csv.DictReader((DATA / "precision_resources.csv").open()))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    color = {3: OCHRE, 4: TEAL, 5: NAVY}
    summary = []
    for m in [3, 4, 5]:
        for window, style in [("sine", "-"), ("tophat", "--")]:
            points = []
            for b in [2, 4, 6, 8]:
                group = [
                    r
                    for r in rows
                    if int(r["phase_bits"]) == m
                    and int(r["magnitude_bits"]) == b
                    and r["window"] == window
                ]
                mean = float(np.mean([float(r["diamond_upper"]) for r in group]))
                work = float(np.mean([float(r["worst_work_leakage"]) for r in group]))
                cx = next(
                    float(r["cx"])
                    for r in costs
                    if int(r["phase_bits"]) == m
                    and int(r["magnitude_bits"]) == b
                    and r["window"] == window
                )
                points.append((b, mean, cx, work))
                summary.append(
                    {
                        "m": m,
                        "b": b,
                        "window": window,
                        "diamond_mean": mean,
                        "cx": cx,
                        "work_mean": work,
                    }
                )
            p = np.array(points)
            label = f"m={m} " + ("sine" if window == "sine" else "uniform")
            axs[0].plot(p[:, 0], p[:, 1], {3:"s",4:"^",5:"o"}[m] + style, color=color[m], label=label)
            axs[1].plot(p[:, 2], p[:, 1], {3:"s",4:"^",5:"o"}[m] + style, color=color[m])
            if window == "sine":
                for b, y, cx, _ in points:
                    if m != 5 and b not in [2, 8]:
                        continue
                    axs[1].annotate(
                        f"b={b}",
                        (cx, y),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=8,
                    )
    axs[0].set(
        xlabel="Magnitude bits b (phase bits m held fixed)",
        ylabel="Mean full diamond error",
        title="Truncation bias is a separate precision bottleneck",
    )
    axs[1].set(
        xlabel="Expanded CX count (level 0, no topology)",
        ylabel="Mean full diamond error",
        title="Register allocation at measured gate cost",
    )
    for ax in axs:
        ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both")
    axs[1].set_xscale("log")
    axs[0].legend(fontsize=8, ncol=2)
    finish(fig,ROOT/"outputs/figures/A_06_bit_allocation")
    (ROOT / "outputs/precision_summary.json").write_text(
        json.dumps(
            {
                "groups": summary,
                "n_cases": len(rows),
                "n_compilations": len(costs),
                "max_sdp_gap": max(float(r["sdp_gap"]) for r in rows),
            },
            indent=2,
        )
    )
    print(
        "Precision results",
        len(rows),
        "max gap",
        max(float(r["sdp_gap"]) for r in rows),
    )


if __name__ == "__main__":
    run()
