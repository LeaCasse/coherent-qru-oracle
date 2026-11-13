"""Workspace renewal: structural budget, channel guarantee and readout."""
import csv,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_theme import setup,finish,NAVY,OCHRE,TEAL,SLATE
ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('QRU_DATA_DIR',ROOT/'reference_data'))

def run():
    rows=list(csv.DictReader((DATA/'renewal_study/policies.csv').open()))
    setup();fig,axes=plt.subplots(1,2,figsize=(8.4,3.15))
    selected=[r for r in rows if r['seed']=='600' and r['phase_bits']=='3' and r['protocol']=='alternating' and r['policy'].startswith('budget')]
    axes[0].plot([int(r['a_calls']) for r in selected],[float(r['certificate']) for r in selected],'s--',color=OCHRE,label='Channel certificate')
    axes[0].plot([int(r['a_calls']) for r in selected],[float(r['diamond_upper']) for r in selected],'o-',color=NAVY,label='Diamond upper witness')
    fresh=next(r for r in rows if r['seed']=='600' and r['phase_bits']=='3' and r['protocol']=='alternating' and r['policy']=='fresh')
    axes[0].scatter([int(fresh['a_calls'])],[float(fresh['diamond_upper'])],marker='D',color=SLATE,label='Fresh work',zorder=5)
    axes[0].set(xlabel='Structural preparation calls',ylabel='Full diamond error',title='(a)  Certified choices')
    for policy,color,marker in [('shared',OCHRE,'s'),('budget6',TEAL,'^'),('fresh',NAVY,'o')]:
        data=sorted([r for r in rows if r['depth']=='2' and r['phase_bits']=='4' and r['protocol']=='alternating' and r['policy']==policy],key=lambda r:int(r['seed']))
        label={'shared':'Shared','budget6':'Selected','fresh':'Fresh'}[policy]
        axes[1].plot(range(1,7),[float(r['readout_error']) for r in data],marker=marker,ls={'shared':'--','budget6':'-.','fresh':'-'}[policy],color=color,label=label)
    axes[1].set(xlabel='Model',ylabel='Input-interference error',title='(b)  Coherent readout')
    for ax in axes:ax.legend(fontsize=7.5);ax.grid(alpha=.15)
    finish(fig,ROOT/'outputs/figures/06_renewal.pdf')

if __name__=='__main__':run()
