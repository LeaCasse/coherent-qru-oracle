"""Fixed score-dependent protocols and clean-work renewal certificates."""
import csv,json,os,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from qru_registerization.memory_contract import clean_contract,shared_bound,partitions,partition_bound,select_partition
from qru_registerization.process import process_channel,choi_from_superoperator,superoperator_from_choi,diamond_choi,rotation
from qru_registerization.qru import random_qru_params

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('QRU_RESULTS_DIR',ROOT/'outputs/results'))/'renewal_study'
XS=[-.7,.9]
AXIS=[.3,-.4,np.sqrt(.75)]
Z=[0,0,1];X=[1,0,0]

def protocol(name):
    axes=[Z]*6 if name=='repeated' else [Z,X,Z,X,Z,X]
    pulses=[(axis,a) for axis,a in zip(axes,[.45,.3,.5,.35,.4,.25])]
    gates=[np.kron(np.eye(2),rotation([0,1,0],a)) for a in [.2,-.3,.15,.25,-.2]]
    if name=='input_mixed':
        h=np.array([[1,1],[1,-1]])/np.sqrt(2)
        gates=[np.kron(h,np.eye(2))@g for g in gates]
    return pulses,gates

def segments(params,m,name):
    pulses,gates=protocol(name);k=len(pulses)
    results={};contracts={};costs={}
    for a in range(k):
        for b in range(a+1,k+1):
            result=process_channel(params,XS,AXIS,pulses[a:b],gates[a:b-1],phase_bits=m)
            results[a,b]=result
            contracts[a,b]=clean_contract(result.zero_kraus,result.ideal)
            # One pair after fusion, or one pair per call when input is mixed.
            costs[a,b]=(b-a if name=='input_mixed' else 1)*(4*2**m-2)
    return results,contracts,costs

def compose(ends,results,gates):
    d=len(next(iter(results.values())).ideal);sp=np.eye(d*d,dtype=complex)
    start=0
    for end in ends:
        sp=superoperator_from_choi(results[start,end].choi)@sp
        if end < ends[-1]:sp=np.kron(gates[end-1].conj(),gates[end-1])@sp
        start=end
    return choi_from_superoperator(sp)

def save(rows,name):
    temp=OUT/(name+'.partial')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,OUT/name)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    spec={'depth_seeds':{'2':list(range(600,606)),'4':list(range(700,703))},'phase_bits':[3,4],
          'protocols':['repeated','alternating','input_mixed'],'inputs':XS,'axis':AXIS,
          'selection':'minimum clean-block certificate within structural-call budget',
          'search':'all 32 contiguous partitions of six calls',
          'cost':'structural A/A-dagger calls with permitted estimator fusion',
          'diamond_subset':'seed 600, m=3, all protocols and selected policies',
          'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/qru_registerization/memory_contract.py']}}
    (OUT/'protocol.json').write_text(json.dumps(spec,indent=2))
    rows=[];bounds=[]
    for depth,seeds in [(2,range(600,606)),(4,range(700,703))]:
        for seed in seeds:
            params=random_qru_params(depth,seed=seed,scale=.6)
            for m in [3,4]:
                for name in spec['protocols']:
                    results,contracts,costs=segments(params,m,name)
                    _,gates=protocol(name);ideal=results[0,6].ideal;v=ideal.ravel(order='F');idealj=np.outer(v,v.conj())
                    generic=shared_bound([contracts[j,j+1] for j in range(6)])
                    bounds.append({'depth':depth,'seed':seed,'phase_bits':m,'protocol':name,'general_bound':generic,'whole_block_bound':contracts[0,6].channel_bound})
                    candidates=[('shared',(6,),contracts[0,6].channel_bound,costs[0,6]),
                                ('fresh',(1,2,3,4,5,6),partition_bound((1,2,3,4,5,6),contracts),sum(costs[j,j+1] for j in range(6)))]
                    unit=4*2**m-2
                    for budget in [1,2,3,4,5,6]:
                        try:ends,cert,cost=select_partition(6,contracts,costs,budget*unit)
                        except ValueError:continue
                        candidates.append((f'budget{budget}',ends,cert,cost))
                    cache={}
                    for policy,ends,cert,cost in candidates:
                        if ends not in cache:
                            j=compose(ends,results,gates);delta=j-idealj
                            witness=float(np.linalg.norm(delta,ord='nuc')/4)
                            # Ramsey readout of the input after a coherent target computation.
                            state=np.kron(np.ones(2)/np.sqrt(2),np.ones(2)/np.sqrt(2));rho=np.outer(state,state.conj())
                            sp=superoperator_from_choi(j);out=(sp@rho.ravel(order='F')).reshape(4,4,order='F')
                            obs=np.kron(np.array([[0,1],[1,0]]),np.eye(2))
                            measured=float(np.trace(obs@out).real);expected=float(np.vdot(ideal@state,obs@(ideal@state)).real)
                            metrics={'choi_error':witness,'readout':measured,'ideal_readout':expected,'readout_error':abs(measured-expected)}
                            if seed==600 and m==3:metrics.update(diamond_choi(delta))
                            else:metrics.update({'diamond_lower':float('nan'),'diamond_upper':float('nan'),'sdp_gap':float('nan')})
                            assert witness <= cert+1e-8
                            if np.isfinite(metrics['diamond_upper']):assert metrics['diamond_upper'] <=cert+1e-5
                            cache[ends]=metrics
                        rows.append({'depth':depth,'seed':seed,'phase_bits':m,'protocol':name,'policy':policy,'ends':'-'.join(map(str,ends)),'segments':len(ends),'a_calls':cost,'certificate':cert,**cache[ends]})
                    save(rows,'policies.csv');save(bounds,'bounds.csv')
                    print(depth,seed,m,name,flush=True)
    summary={'policy_rows':len(rows),'bound_rows':len(bounds),
             'selected_strictly_better_than_both_endpoints':sum(r['policy']=='budget6' and r['choi_error'] < min(s['choi_error'] for s in rows if all(r[k]==s[k] for k in ['depth','seed','phase_bits','protocol']) and s['policy'] in ['shared','fresh'])-1e-10 for r in rows)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)

if __name__=='__main__':run()
