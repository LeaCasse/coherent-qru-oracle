"""Explicit quaternion lift and invariant physical score under a right fibre."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_theme import setup,finish,NAVY,OCHRE
setup()
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from qru_registerization.qru import evaluate_trajectory, random_qru_params, rz
from qru_registerization.quaternion_geometry import (
    bloch_from_quaternion,
)
from qru_registerization.quaternion_lift import su2_lift_quaternion

ROOT = Path(__file__).resolve().parents[1]


def run():
    xs = np.linspace(-0.73, 0.91, 121)
    p = random_qru_params(2, seed=200, scale=0.6)
    path = evaluate_trajectory(p, xs)
    phases = 1.3 * (xs - xs[0]) / (xs[1] - xs[0]) * 3 / 120
    changed = np.array(
        [
            su2_lift_quaternion(u @ rz(phi))
            for u, phi in zip(path.unitaries, phases, strict=True)
        ]
    )
    c = np.cos(phases / 2)
    s = np.sin(phases / 2)
    q = np.array([su2_lift_quaternion(u) for u in path.unitaries])
    algebra = np.column_stack(
        [
            q[:, 0] * c - q[:, 3] * s,
            q[:, 1] * c + q[:, 2] * s,
            q[:, 2] * c - q[:, 1] * s,
            q[:, 3] * c + q[:, 0] * s,
        ]
    )
    np.testing.assert_allclose(algebra, changed, atol=1e-12)
    rb = np.array([bloch_from_quaternion(v) for v in changed])
    axis = np.array([0.3, -0.4, np.sqrt(0.75)])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].plot(xs, q[:, 0], color=NAVY,label="Original $q_0$")
    axs[0].plot(xs, changed[:, 0], "--", color=OCHRE,label="Right-fibre $q_0$")
    axs[0].set(
        xlabel="Input x",
        ylabel="Scalar quaternion coordinate",
        title=r"Lift changes under $q \mapsto q r_z(\phi)$",
    )
    axs[1].plot(xs, path.bloch @ axis, color=NAVY,lw=2.4,marker="o",markevery=20,ms=4,label="Original physical score")
    axs[1].plot(xs, rb @ axis, "--",color=OCHRE,lw=1.2,marker="s",markevery=(10,20),ms=4,label="Right-fibre score")
    axs[1].set(
        xlabel="Input x",
        ylabel=r"Oriented score $v\cdot r$",
        title="Physical score remains invariant",
    )
    for ax in axs:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    finish(fig,ROOT/"outputs/figures/A_07_quaternion_fibre")
    print(
        "Fibre quaternion identity checked; Bloch difference",
        np.max(abs(rb - path.bloch)),
    )


if __name__ == "__main__":
    run()
