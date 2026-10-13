"""Fresh discarded workspace vs coherent reuse of the same dirty workspace."""

import csv
import os
from pathlib import Path

import numpy as np

from qru_registerization.channel import coherent_channel, schur_diamond
from qru_registerization.qru import random_qru_params

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QRU_RESULTS_DIR", ROOT / "outputs/results")) / "direct_control"


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(400,406):
        params = random_qru_params(2, seed=seed, scale=0.6)
        for m in [3,5]:
            n=2**m
            scores=-np.cos(2*np.pi*np.arange(n)/n)/np.cos(np.pi/n)
            kwargs={"phase_bits":m,"taper":"sine","control_scores":scores}
            base=coherent_channel(
                params, np.linspace(-.73,.91,4), [.3,-.4,np.sqrt(.75)], **kwargs,
            )
            per_step=schur_diamond(
                base.gram-np.outer(base.ideal,base.ideal.conj()),
            )["diamond_upper"]
            for k in [1,2,4,8,16,32]:
                ideal=np.outer(base.ideal**k,(base.ideal**k).conj())
                fresh=base.gram**k  # Schur product, NOT matrix multiplication
                reused=coherent_channel(
                    params, np.linspace(-.73,.91,4), [.3,-.4,np.sqrt(.75)],
                    gamma=k*.8, **kwargs,
                )
                for mode,gram in [("fresh",fresh),("same_work",reused.gram)]:
                    rows.append({
                        "seed":seed,"phase_bits":m,"repetitions":k,"mode":mode,
                        **schur_diamond(gram-ideal),
                        "fresh_composition_bound":min(2,k*per_step),
                    })
                with (OUT/"reuse.csv").open("w",newline="") as file:
                    writer=csv.DictWriter(file,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        print(f"Reuse: seed {seed}",flush=True)


if __name__=="__main__":
    run()
