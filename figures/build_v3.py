"""Render V3 figures from the executed, unfiltered result tables."""
from pathlib import Path
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('QRU_V3_DATA_DIR',ROOT/'reference_data/v3'));OUT=ROOT/'outputs/figures/v3'
COLORS=['#234f75','#c46a27','#207c6e']
POLICIES=[('fresh','Fresh work'),('shared','Shared work'),('shared_phase_pi','Shared + workspace Z')]

def read(name):
    path=ROOT/'reference_data/channel_study/precision.csv' if name=='legacy_precision.csv' else DATA/name
    return list(csv.DictReader(path.open()))
def val(rows,key):return np.array([float(r[key]) for r in rows])
def finish(fig,name):
    for ax in fig.axes:ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(pad=1.2)
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=160)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.7,2.65))
    rows=read('direct_memory.csv')
    for (policy,label),color in zip(POLICIES,COLORS):
        xs=[2,4,8,16,32];ys=[]
        for k in xs:
            group=[r for r in rows if r['policy']==policy and r['total_angle']=='0.8' and r['protocol']=='alternating' and int(r['uses'])==k]
            assert len(group)==3
            ys.append(val(group,'choi_error').mean())
        axes[0].plot(xs,ys,'o-',label=label,color=color,ms=4)
    axes[0].set(xscale='log',yscale='log',xlabel='Calls',ylabel='Normalized Choi witness',title='(a) Coherent reuse')
    axes[0].legend(fontsize=8)
    rows=read('direct_memory_validation.csv')
    for (policy,label),color in zip(POLICIES,COLORS):
        group=sorted([r for r in rows if r['policy']==policy and r['total_angle']=='0.8' and r['protocol']=='alternating' and r['uses']=='32'],key=lambda r:int(r['seed']))
        assert len(group)==7
        axes[1].plot(range(1,8),val(group,'choi_error'),'o-',color=color,ms=4,label=label)
    axes[1].set(yscale='log',xlabel='New evaluation model',title='(b) Independent models',xticks=[1,3,5,7])
    finish(fig,'v3_memory')

    fig,axes=plt.subplots(1,2,figsize=(6.7,2.65))
    rows=read('amortization_resources.csv');positions=np.arange(2);width=.32
    for i,(method,label) in enumerate([('shared_numeric','Shared numeric QPE'),('modular_gqsp_fresh','Modular fresh GQSP')]):
        group=[r for r in rows if r['method']==method and r['uses']=='32' and r['input_bits']=='2' and r['optimization_level']=='3']
        group=sorted(group,key=lambda r:int(r['depth']))
        assert len(group)==2
        axes[0].bar(positions+(i-.5)*width,val(group,'cx'),width,label=label,color=COLORS[i])
    axes[0].set(xticks=positions,xticklabels=['2','16'],xlabel='Preparation depth',ylabel='Compiled CX',title='(a) Modular sequence')
    axes[0].legend(fontsize=8)
    rows=read('renewal_scaling.csv')
    group=sorted([r for r in rows if r['branches']=='16' and r['phase_bits']=='5' and r['target_dimension']=='4' and r['regime']=='fixed_total'],key=lambda r:int(r['uses']))
    axes[1].plot(val(group,'uses'),val(group,'whole_certificate'),'o-',label='One shared segment',color=COLORS[0],ms=4)
    axes[1].plot(val(group,'uses'),val(group,'selected_certificate'),'s-',label='Selected renewal',color=COLORS[2],ms=4)
    axes[1].set(yscale='log',xlabel='Calls',ylabel='Channel certificate',title='(b) Target dimension four')
    axes[1].legend(fontsize=8)
    finish(fig,'v3_design')

    fig,ax=plt.subplots(figsize=(5.6,2.8));rows=read('fibre.csv')
    for init,label,color in [('prepared','Prepared',COLORS[0]),('zero','Invariant',COLORS[2])]:
        group=sorted([r for r in rows if r['initialization']==init and r['protocol']=='mixed' and r['phase_bits']=='3' and r['uses']=='4'],key=lambda r:int(r['seed']))
        assert len(group)==9
        ax.plot(range(1,10),val(group,'fibre_choi_sensitivity'),'o-',label=label,color=color,ms=4)
    ax.set(xlabel='Evaluation model',ylabel='Normalized Choi sensitivity',title='Input mixing');ax.legend(fontsize=8)
    finish(fig,'v3_geometry')

    fig,ax=plt.subplots(figsize=(5.6,2.8));rows=read('inverse.csv')
    for mode,label,color in [('paired','Matched error',COLORS[2]),('mismatch','Inverse mismatch',COLORS[1]),('drift','Call-dependent drift',COLORS[0])]:
        xs=[1e-4,1e-3,.01,.05];ys=[]
        for delta in xs:
            group=[r for r in rows if r['error_model']==mode and r['mode']=='shared' and r['uses']=='8' and r['protocol']=='echo' and float(r['delta'])==delta]
            assert len(group)==5
            ys.append(max(1e-14,val(group,'choi_error').mean()))
        ax.plot(xs,ys,'o-',label=label,color=color,ms=4)
    ax.set(xscale='log',yscale='log',xlabel='Perturbation strength',ylabel='Normalized Choi witness',title='Eight-call echo');ax.legend(fontsize=7)
    finish(fig,'v3_inverse')

    fig,ax=plt.subplots(figsize=(5.6,2.8));rows=read('precision.csv')
    for method,label,color in [('qpe_prepared','Numeric QPE',COLORS[0]),('gqsp','GQSP',COLORS[1])]:
        xs=[];ys=[];lower=[];upper=[]
        for eps in [.05,.01,.002]:
            group=[r for r in rows if r['method']==method and r['gamma']=='0.8' and r['contract']=='channel_only' and float(r['tolerance'])==eps]
            assert len(group)==9
            errors=val(group,'diamond_upper');xs.append(float(group[0]['a_calls']));ys.append(errors.mean())
            lower.append(errors.mean()-errors.min());upper.append(errors.max()-errors.mean())
        ax.errorbar(xs,ys,yerr=[lower,upper],fmt='o-',capsize=3,label=label,color=color,ms=4)
    ax.set(xscale='log',yscale='log',xlabel='Preparation / inverse calls',ylabel='Full diamond upper bound',title='Single rotation');ax.legend(fontsize=8)
    finish(fig,'v3_precision')

    fig,ax=plt.subplots(figsize=(5.6,2.8));rows=read('legacy_precision.csv')
    for m,color,marker in [(3,COLORS[1],'s'),(4,COLORS[2],'^'),(5,COLORS[0],'o')]:
        for window,style in [('sine','-'),('tophat','--')]:
            ys=[]
            for b in [2,4,6,8]:
                group=[r for r in rows if int(r['phase_bits'])==m and int(r['magnitude_bits'])==b and r['window']==window]
                assert len(group)==6
                ys.append(val(group,'diamond_upper').mean())
            label=f'm={m}, '+('sine' if window=='sine' else 'uniform')
            ax.plot([2,4,6,8],ys,marker+style,color=color,label=label,ms=4)
    ax.set(yscale='log',xlabel='Magnitude bits',ylabel='Mean full diamond error',title='Numerical allocation')
    ax.legend(fontsize=7,ncol=2)
    finish(fig,'v3_bits')

if __name__=='__main__':main()
