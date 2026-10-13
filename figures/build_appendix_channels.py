"""Rebuild three appendix figures from unchanged archived CSV tables."""
import os
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_theme import setup,finish,NAVY,OCHRE,TEAL,SLATE,CHARCOAL
ROOT=Path(__file__).resolve().parents[1]
setup()
def read(name):
 with (Path(os.environ.get('QRU_DATA_DIR', ROOT/'reference_data'))/'channel_study'/name).open() as f:return list(csv.DictReader(f))
def values(rows,key):return np.array([float(r[key]) for r in rows])
def bootstrap(a):
 a=np.asarray(a);boot=np.mean(np.random.default_rng(917).choice(a,(5000,len(a))),axis=1)
 return [float(np.mean(a)),*np.quantile(boot,[.025,.975]).tolist()]
def save(fig,name):finish(fig,ROOT/'outputs/figures'/('A_'+name))
def run():
    rows=read('channels.csv');gauge=read('gauge.csv');noise=read('dephasing.csv')
    summary={'paired_windows':[]}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for clean, color in [("0", OCHRE), ("1", NAVY)]:
        means = []
        for m in range(2, 6):
            group = [
                r for r in gauge if r["cleanup"] == clean and int(r["phase_bits"]) == m
            ]
            means.append(np.mean(values(group, "diamond_lower")))
            axs[0].scatter(
                np.full(len(group), m),
                values(group, "diamond_lower"),
                color=color,
                alpha=0.4,
                marker="s" if clean=="0" else "o",
                s=18,
            )
        axs[0].plot(
            range(2, 6),
            means,
            "s--" if clean=="0" else "o-",
            color=color,
            label="Without inverse" if clean == "0" else "With inverse",
        )
    group = [r for r in gauge if r["cleanup"] == "1"]
    axs[1].scatter(
        values(group, "system_one_bound"),
        values(group, "diamond_upper"),
        color=NAVY,
        s=35,
    )
    limit = max(values(group, "system_one_bound")) * 1.1
    axs[1].plot([0, limit], [0, limit], "k--", label="Bound $2p_1$")
    axs[0].set(
        xlabel="Phase bits m",
        ylabel="Diamond change under fibre",
        title="Same fixed-input scores, different lifts",
    )
    axs[1].set(
        xlabel="Twice worst system-one probability",
        ylabel="Diamond change after inverse",
        title="Fibre survives only in the leakage sector",
    )
    for ax in axs:
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    save(fig, "02_fibre_cancellation")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for split, ls in [("test", "-"), ("shift_depth", "--")]:
        for key, ax in zip(
            ["fixed_input_scalar_loss", "diamond_lower"], axs, strict=True
        ):
            ratios = []
            for m in range(2, 6):
                sine = {
                    int(r["seed"]): float(r[key])
                    for r in rows
                    if r["split"] == split
                    and r["window"] == "sine"
                    and int(r["phase_bits"]) == m
                }
                learned = {
                    int(r["seed"]): float(r[key])
                    for r in rows
                    if r["split"] == split
                    and r["window"] == "learned_scalar"
                    and int(r["phase_bits"]) == m
                }
                paired = np.array([learned[s] - sine[s] for s in sorted(sine)])
                ci = bootstrap(paired)
                ratio = np.mean(list(learned.values())) / np.mean(list(sine.values()))
                ratios.append(ratio)
                summary["paired_windows"].append(
                    {
                        "split": split,
                        "m": m,
                        "metric": key,
                        "learned_minus_sine_mean_ci": ci,
                        "ratio_of_means": float(ratio),
                    }
                )
            ax.plot(
                range(2, 6),
                ratios,
                ("o" if split=="test" else "s") + ls,
                color=NAVY if split=="test" else OCHRE,
                label="L=2 held out" if split == "test" else "L=4 shift",
            )
            ax.axhline(1, color="black", lw=0.8)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Phase bits m")
    axs[0].set(
        ylabel="Ratio of mean losses (learned / sine)",
        title="Objective used to train the window",
    )
    axs[1].set(
        ylabel="Ratio of mean diamond errors",
        title="Whole-channel criterion, not optimized",
    )
    axs[0].legend()
    save(fig, "03_scalar_vs_channel")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, gamma in zip(axs, [0.0, 0.8], strict=True):
        for m in [2, 3, 4, 5]:
            group = [
                r
                for r in noise
                if float(r["gamma"]) == gamma
                and r["cleanup"] == "1"
                and int(r["phase_bits"]) == m
            ]
            ax.plot(
                values(group, "dephase"),
                values(group, "diamond_upper"),
                {2:"o-",3:"s--",4:"^-.",5:"D:"}[m],
                color={2:NAVY,3:OCHRE,4:TEAL,5:SLATE}[m],
                label=f"m={m}",
            )
        base = [
            r
            for r in noise
            if float(r["gamma"]) == gamma
            and r["cleanup"] == "0"
            and int(r["phase_bits"]) == 5
        ]
        ax.plot(
            values(base, "dephase"),
            values(base, "diamond_upper"),
            color=CHARCOAL,linestyle=(0,(5,2,1,2)),marker="v",
            label="No inverse, m=5",
        )
        ax.set(
            xlabel="Probability of phase-word dephasing",
            title=f"gamma = {gamma:g}, seed 200",
        )
        ax.grid(alpha=0.2)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.065),ncol=5,fontsize=8)
    axs[0].set_ylabel("Full diamond error to ideal oracle")
    save(fig, "04_dephasing_intervention")


if __name__=="__main__":run()
