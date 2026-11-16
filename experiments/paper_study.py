"""V3 campaigns with recorded configurations and separate output tables.

Run: PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python experiments/paper_study.py --stage all
No experiment discards a model because an architecture performs poorly.
"""
import argparse,csv,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag,expm
from qiskit import transpile,QuantumCircuit
from qiskit.circuit.library import UCRZGate,UnitaryGate
from qru_registerization.qru import random_qru_params,qru_unitary,signed_coordinate,rz
from qru_registerization.interface import directional_amplitude_unitary,phase_taper_amplitudes
from qru_registerization.channel import estimator_matrix
from qru_registerization.process import process_channel,process_metrics,kraus_choi,diamond_choi,rotation
from qru_registerization.structural import optimized_score_service,affine_estimator
from qru_registerization.memory_contract import select_partition,select_partition_exhaustive,clean_contract,shared_bound
from qru_registerization.certification import preserved_segment_certificates,uniform_moment_certificate,sine_variance_bound,generator_rotations
from qru_registerization.numeric_service import fused_numeric_kraus,choose_phase_bits,target_protocol
from qru_registerization.direct_qsp import make_gqsp_plan,branch_dilation,gqsp_score_service
from qru_registerization.dilation_tools import branch_kraus,sequence_channel,channel_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/results/v3'
AXIS=[.3,-.4,np.sqrt(.75)]
Z=np.diag([1.,-1.]);X=np.array([[0.,1.],[1.,0.]]);Y=np.array([[0.,-1j],[1j,0.]])
MODELS=[(2,s) for s in range(900,906)]+[(4,s) for s in range(1000,1003)]
SPEC={
 'status':'declared before executing the new main campaigns; not an external preregistration',
 'development_seeds':[400,603,604], 'evaluation_models':MODELS,
 'precision':{'gamma':[.2,.8,1.6],'channel_tolerance':[.05,.01,.002],'input_branches':4,
              'extra_literal_contract':{'channel_tolerance':.01,'literal_tolerance':.01}},
 'fibre':{'phase_bits':[2,3,4],'uses':[2,4,8],'protocols':['repeated','echo','alternating','mixed'],
          'phase_change':[0.,2.1],'initializations':['prepared','zero']},
 'scaling':{'uses':[8,16,32,64],'branches':[2,16],'phase_bits':[3,5],
            'target_dimensions':[2,4],'angular_budget_regimes':['fixed_total','fixed_per_call']},
 'inverse':{'models':[(2,900),(2,901),(2,902),(4,1000),(4,1001)],'phase_bits':3,
            'perturbation_generator':'Pauli X on the least-significant phase-word qubit after the estimator',
            'uses':[2,8],'delta':[0.,.0001,.001,.01,.05],
            'error_models':['paired','mismatch','drift'],'protocols':['echo','alternating'],'policies':['shared','fresh']},
 'amortization':{'models':[(2,900),(2,901),(4,1000)],'uses':[2,4,8,16,32],
                 'intervention_angle_budget':.4,
                 'total_angles':[.8,2.4],'channel_tolerance':.01,
                 'comparison':'one shared numeric estimate versus modular per-call direct GQSP with fresh work; not globally optimized QSP'},
 'amortization_resources':{'depths':[2,4,16],'uses':[2,8,32],'total_angle':.8,
                           'tolerance':.01,'intervention_angle_budget':.4,'input_bits':[1,2],
                           'status':'exploratory input-dimension ablation after observing small-space compilation compression at r=1'},
 'direct_memory':{'status':'mechanistic extension declared after single-use and amortization campaigns',
                  'models':[(2,900),(2,901),(4,1000)],'uses':[2,4,8,16,32],'total_angles':[.8,2.4],
                  'protocols':['repeated','alternating','echo','mixed'],
                  'policies':['fresh','shared','shared_phase_pi'],
                  'phase_kick':'Pauli Z on the GQSP output ancilla after every call, including the last',
                  'purpose':'same isolated channel and local contracts, different coherent reuse; not an adaptive phase optimization'},
 'direct_memory_validation':{'status':'new-seed validation declared after the direct-memory discovery',
                            'models':[(2,s) for s in range(1500,1504)]+[(4,s) for s in range(1600,1603)],
                            'uses':[16,32],'total_angles':[.8,1.2,2.4],
                            'protocols':['repeated','alternating','mixed'],'policies':['fresh','shared','shared_phase_pi']},
 'metrics':{'diamond':'full norm, no factor one half','choi':'normalized trace-norm witness; never relabelled diamond',
            'literal':'operator norm of actual zero-work isometry minus chosen clean ideal isometry'},
 'compilation':{'basis':['rz','sx','x','cx'],'seed':17,'main_optimization_level':0,'selected_level':3,'coupling_map':None},
 'scope':'single-qubit QRU scores; independent target, ideal gates except explicit inverse perturbations; no advantage claim over classical reversible score computation',
 'selection':'all models and policies retained; synthesis failures stop the campaign rather than silently dropping a case'}


def write(rows,name):
    OUT.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    temporary=OUT/(name+'.partial')
    with temporary.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    temporary.replace(OUT/name)


def metrics(ks,ideal,diamond=False):
    return channel_metrics(kraus_choi(ks),ks[0],ideal,ks,diamond)


def qsp_kraus(params,xs,gamma,tolerance,literal=None):
    plan=make_gqsp_plan(float(gamma),float(tolerance),literal)
    amps=[directional_amplitude_unitary(qru_unitary(params,float(x)),AXIS) for x in xs]
    ks=branch_kraus([branch_dilation(a,plan) for a in amps],2)
    scores=np.array([signed_coordinate(qru_unitary(params,float(x))[:,0],AXIS) for x in xs])
    ideal=block_diag(*generator_rotations(Z,gamma,scores))
    return ks,ideal,plan


def precision():
    rows=[];xs=np.linspace(-.73,.91,4)
    for depth,seed in MODELS:
        params=random_qru_params(depth,seed=seed,scale=.6)
        targets=[(g,eps,None) for g in SPEC['precision']['gamma'] for eps in SPEC['precision']['channel_tolerance']]
        targets.append((.8,.01,.01))
        for gamma,tolerance,literal in targets:
            m=choose_phase_bits(gamma,tolerance,literal)
            common={'depth':depth,'seed':seed,'gamma':gamma,'tolerance':tolerance,'literal_tolerance':literal or '',
                    'contract':'channel_only' if literal is None else 'channel_and_literal'}
            for init in ['prepared','zero']:
                start=time.perf_counter()
                ks,ideal,_=fused_numeric_kraus(params,xs,AXIS,m,[(Z,gamma)],initialization=init)
                data=metrics(ks,ideal,diamond=True)
                uniform=2*gamma**2*sine_variance_bound(m)
                assert data['diamond_upper']<=uniform+2e-6
                assert data['literal_error']<=abs(gamma)*np.sqrt(sine_variance_bound(m))+1e-9
                rows.append(common|{'method':'qpe_'+init,'phase_bits':m,'degree':'','a_calls':4*2**m-(2 if init=='prepared' else 4),
                   'qubits':2+m+3,'uniform_certificate':uniform,'synthesis_seconds':0.,'evaluation_seconds':time.perf_counter()-start,
                   'synthesis_coefficient_residual':'',**data})
            start=time.perf_counter();ks,ideal,plan=qsp_kraus(params,xs,gamma,tolerance,literal)
            data=metrics(ks,ideal,diamond=True)
            assert data['diamond_upper']<=plan.channel_error_upper+2e-6
            assert data['literal_error']<=plan.literal_error_upper+1e-9
            rows.append(common|{'method':'gqsp','phase_bits':'','degree':plan.degree,'a_calls':plan.preparation_calls,
                'qubits':5,'uniform_certificate':plan.channel_error_upper,'synthesis_seconds':plan.synthesis_seconds,
                'evaluation_seconds':time.perf_counter()-start,'synthesis_coefficient_residual':plan.coefficient_residual,**data})
        write(rows,'precision.csv');print('precision',depth,seed,len(rows),flush=True)


def resources():
    rows=[];plans={}
    for depth,seed in [(2,900),(4,1000)]:
        params=random_qru_params(depth,seed=seed,scale=.6)
        targets=[(g,eps,None) for g in [.2,.8,1.6] for eps in [.05,.01,.002]]+[(.8,.01,.01)]
        for gamma,tolerance,literal in targets:
            m=choose_phase_bits(gamma,tolerance,literal);n=2**m
            table=-np.cos(2*np.pi*np.arange(n)/n)/np.cos(np.pi/n)
            plan=make_gqsp_plan(float(gamma),float(tolerance),literal)
            plans[f'{gamma}/{tolerance}/{literal}']=plan.metadata()|{'angles':plan.angles.tolist()}
            circuits=[]
            for init in ['prepared','zero']:
                c,_=optimized_score_service(params,2,-.73,1.64/3,AXIS,phase_bits=m,gamma=gamma,taper='sine',control_scores=table,initialization=init)
                circuits.append(('qpe_'+init,c,4*n-(2 if init=='prepared' else 4)))
            c,_=gqsp_score_service(params,2,-.73,1.64/3,AXIS,plan)
            circuits.append(('gqsp',c,plan.preparation_calls))
            levels=[0,3] if gamma==.8 and tolerance==.01 and literal is None else [0]
            for level in levels:
                for method,c,calls in circuits:
                    start=time.perf_counter();compiled=transpile(c,basis_gates=['rz','sx','x','cx'],optimization_level=level,seed_transpiler=17)
                    rows.append({'depth':depth,'seed':seed,'gamma':gamma,'tolerance':tolerance,'literal_tolerance':literal or '',
                       'method':method,'optimization_level':level,'a_calls':calls,'qubits':c.num_qubits,'cx':compiled.count_ops().get('cx',0),
                       'compiled_depth':compiled.depth(),'rz':compiled.count_ops().get('rz',0),'compile_seconds':time.perf_counter()-start})
            write(rows,'resources.csv')
        print('resources',depth,len(rows),flush=True)
    (OUT/'gqsp_synthesis.json').write_text(json.dumps(plans,indent=2))


def protocol(name,k):
    if name=='repeated':return [([0,0,1],.8)]*k,[np.eye(4)]*(k-1)
    if name=='echo':return [([0,0,1],.8)]*k,[np.kron(np.eye(2),X)]*(k-1)
    if name=='alternating':return [([0,0,1] if j%2==0 else [1,0,0],.8) for j in range(k)],[np.eye(4)]*(k-1)
    had=np.array([[1,1],[1,-1]])/np.sqrt(2)
    return [([0,0,1],.8)]*k,[np.kron(had,X)]*(k-1)


def fibre():
    rows=[]
    for depth,seed in MODELS:
        params=random_qru_params(depth,seed=seed,scale=.6)
        for m in [2,3,4]:
            for k in [2,4,8]:
                for name in ['repeated','echo','alternating','mixed']:
                    pulses,gates=protocol(name,k)
                    for init in ['prepared','zero']:
                        base=process_channel(params,[-.7,.9],AXIS,pulses,gates,phase_bits=m,initialization=init)
                        changed=process_channel(params,[-.7,.9],AXIS,pulses,gates,phase_bits=m,initialization=init,fibre=[0,2.1])
                        delta=changed.choi-base.choi
                        sensitivity=float(np.linalg.norm(delta,ord='nuc')/4)
                        p1=float(np.linalg.eigvalsh(sum(q.conj().T@q for q in base.kraus[1::2]))[-1])
                        row={'depth':depth,'seed':seed,'phase_bits':m,'uses':k,'protocol':name,'initialization':init,
                            'fibre_choi_sensitivity':sensitivity,'sector_quantity':2*p1,'sector_bound_applicable':name!='mixed',
                            **process_metrics(base,diamond=False)}
                        if seed in [900,1000] and m==3 and k==4:
                            row|={'fibre_'+key:value for key,value in diamond_choi(delta).items()}
                        if init=='zero':assert sensitivity<1e-10
                        if name=='echo':assert row['choi_trace_norm']<1e-10
                        rows.append(row)
        write(rows,'fibre.csv');print('fibre',depth,seed,len(rows),flush=True)


def renewal():
    from experiments.renewal_study import protocol as old_protocol,segments
    rows=[];max_residual=0.;comparisons=[]
    for depth,seeds in [(2,range(600,606)),(4,range(700,703))]:
        for seed in seeds:
            params=random_qru_params(depth,seed=seed,scale=.6)
            scores=np.array([signed_coordinate(qru_unitary(params,x)[:,0],AXIS) for x in [-.7,.9]])
            for m in [3,4]:
                for name in ['repeated','alternating']:
                    old_pulses,old_gates=old_protocol(name)
                    pulses=[(Z if np.allclose(axis,[0,0,1]) else X,g) for axis,g in old_pulses]
                    gates=[r[:2,:2] for r in old_gates]
                    start=time.perf_counter();cs,ds=preserved_segment_certificates(scores,m,pulses,gates);fast_time=time.perf_counter()-start
                    start=time.perf_counter();dense,old,cost=segments(params,m,name);dense_time=time.perf_counter()-start
                    for ab in cs:
                        max_residual=max(max_residual,abs(cs[ab].clean_error-old[ab].clean_error),abs(cs[ab].leakage-old[ab].leakage))
                    unit=4*2**m-2
                    for budget in range(1,7):
                        chosen,bound,used=select_partition(6,cs,cost,budget*unit)
                        oracle,exact_bound,exact_cost=select_partition_exhaustive(6,old,cost,budget*unit)
                        assert abs(bound-exact_bound)<1e-10
                        comparisons.append({'depth':depth,'seed':seed,'phase_bits':m,'protocol':name,'budget_pairs':budget,
                           'ends':'-'.join(map(str,chosen)),'legacy_ends':'-'.join(map(str,oracle)),
                           'certificate':bound,'legacy_certificate':exact_bound,'a_calls':used,'resets':len(chosen)-1,
                           'fast_certificate_seconds':fast_time,'dense_certificate_seconds':dense_time})
            print('renewal validation',depth,seed,flush=True)
    write(comparisons,'renewal_validation.csv')
    for k in [8,16,32,64]:
        for branches in [2,16]:
            params=random_qru_params(2,seed=900,scale=.6);xs=np.linspace(-.7,.9,branches)
            scores=np.array([signed_coordinate(qru_unitary(params,float(x))[:,0],AXIS) for x in xs])
            for m in [3,5]:
                for t in [2,4]:
                    hs=[Z,X] if t==2 else [np.kron(Z,np.eye(2)),np.kron(X,X)]
                    gate=expm(-.15j*(Y if t==2 else np.kron(Y,Y)))
                    gates=[gate]*(k-1)
                    for regime in ['fixed_total','fixed_per_call']:
                        angle=2.4/k if regime=='fixed_total' else .3
                        pulses=[(hs[j%2],angle) for j in range(k)]
                        start=time.perf_counter();cs,ds=preserved_segment_certificates(scores,m,pulses,gates);cert_seconds=time.perf_counter()-start
                        costs={ab:4*2**m-2 for ab in cs}
                        start=time.perf_counter();ends,bound,used=select_partition(k,cs,costs,min(k,6)*(4*2**m-2));dp_seconds=time.perf_counter()-start
                        uniform=uniform_moment_certificate(m,pulses,gates,grid_points=129)
                        whole=ds[0,k]
                        # A posteriori channel only for a predeclared small subset.
                        actual='';sdp=''
                        if branches==2 and k in [8,32] and m==5:
                            ks,ideal,_=fused_numeric_kraus(params,xs,AXIS,m,pulses,gates)
                            actual=metrics(ks,ideal,diamond=False)['choi_error']
                            assert actual<=whole.contract.channel_bound+1e-9
                            if k==8 and regime=='fixed_total' and t==2:
                                sdp=metrics(ks,ideal,diamond=True)['diamond_upper']
                        rows.append({'uses':k,'branches':branches,'phase_bits':m,'target_dimension':t,'regime':regime,
                            'segments_evaluated':len(cs),'selected_segments':len(ends),'a_calls':used,'resets':len(ends)-1,
                            'certificate_seconds':cert_seconds,'dp_seconds':dp_seconds,'selected_certificate':bound,
                            'whole_certificate':whole.contract.channel_bound,'whole_uniform_certificate':uniform['certificate'],
                            'whole_literal_error':whole.literal_isometry_error,'whole_choi_error':actual,'whole_diamond_upper':sdp})
                        write(rows,'renewal_scaling.csv')
                        del ds
            write(rows,'renewal_scaling.csv');print('renewal scaling',k,branches,len(rows),flush=True)
    (OUT/'renewal_validation.json').write_text(json.dumps({'max_branch_contract_residual':max_residual,'comparisons':len(comparisons)},indent=2))


def inverse():
    rows=[];m=3;n=2**m;values=-np.cos(2*np.pi*np.arange(n)/n)/np.cos(np.pi/n)
    env=2*n;word_flip=np.eye(n)[np.arange(n)^1];generator=np.kron(word_flip,np.eye(2))
    for depth,seed in SPEC['inverse']['models']:
        params=random_qru_params(depth,seed=seed,scale=.6)
        estimators=[estimator_matrix(directional_amplitude_unitary(qru_unitary(params,x),AXIS),phase_taper_amplitudes(m,'sine')) for x in [-.7,.9]]
        scores=np.array([signed_coordinate(qru_unitary(params,x)[:,0],AXIS) for x in [-.7,.9]])
        for k in [2,8]:
            for name in ['echo','alternating']:
                axes,interventions=protocol(name,k)
                ideals=[block_diag(*generator_rotations(Z if axis==[0,0,1] else X,gamma,scores)) for axis,gamma in axes]
                vs=[block_diag(*[generator_rotations(Z if axis==[0,0,1] else X,gamma,np.array([g]))[0] for g in values for _ in range(2)]) for axis,gamma in axes]
                reference=[np.array([np.kron(e.conj().T,np.eye(2))@v@np.kron(e,np.eye(2)) for e in estimators]) for v in vs]
                for delta in SPEC['inverse']['delta']:
                    for error in ['paired','mismatch','drift']:
                        ws=[];etas=[]
                        for j,v in enumerate(vs):
                            perturb=expm(-1j*delta*(1. if error!='drift' else (j+1)/k)*generator)
                            branches=[]
                            for e in estimators:
                                forward=perturb@e if error!='mismatch' else e
                                backward=forward.conj().T if error!='mismatch' else e.conj().T@perturb
                                branches.append(np.kron(backward,np.eye(2))@v@np.kron(forward,np.eye(2)))
                            ws.append(np.array(branches))
                            etas.append(max(np.linalg.norm(a-b,2) for a,b in zip(ws[-1],reference[j])))
                        for mode in ['shared','fresh']:
                            base=sequence_channel(reference,ideals,2,interventions,mode)
                            result=sequence_channel(ws,ideals,2,interventions,mode)
                            data=channel_metrics(*result[:3],kraus=result[3],diamond=False)
                            increment=float(np.linalg.norm(result[0]-base[0],ord='nuc')/4)
                            perturb_bound=min(2.,2*sum(etas))
                            assert increment<=perturb_bound+1e-9
                            if name=='echo' and error=='paired' and mode=='shared':assert data['choi_error']<1e-10
                            row={'depth':depth,'seed':seed,'phase_bits':m,'uses':k,'protocol':name,'error_model':error,
                                  'delta':delta,'mode':mode,'noise_increment_choi':increment,'dilation_perturbation_bound':perturb_bound,**data}
                            if seed==900 and k==8 and delta in [.001,.05]:row.update(diamond_choi(result[0]-base[0]))
                            rows.append(row)
        write(rows,'inverse.csv');print('inverse',depth,seed,len(rows),flush=True)


def amortization():
    rows=[]
    for depth,seed in SPEC['amortization']['models']:
        params=random_qru_params(depth,seed=seed,scale=.6);xs=[-.7,.9]
        scores=np.array([signed_coordinate(qru_unitary(params,x)[:,0],AXIS) for x in xs])
        amps=[directional_amplitude_unitary(qru_unitary(params,x),AXIS) for x in xs]
        for total in [.8,2.4]:
            m=choose_phase_bits(total,.01)
            for k in [2,4,8,16,32]:
                gamma=total/k;pulses=[(Z if j%2==0 else X,gamma) for j in range(k)]
                gates=[expm(-.4j*Y/k)]*(k-1)
                ks,ideal,_=fused_numeric_kraus(params,xs,AXIS,m,pulses,gates)
                numeric=metrics(ks,ideal,diamond=seed==900 and k in [2,32])
                plan=make_gqsp_plan(float(gamma),.01/k)
                zsteps=[branch_dilation(a,plan) for a in amps]
                frame=np.kron(np.eye(4),np.array([[1,1],[1,-1]])/np.sqrt(2))
                steps=[np.array(zsteps) if j%2==0 else np.array([frame@w@frame for w in zsteps]) for j in range(k)]
                ideals=[block_diag(*generator_rotations(h,g,scores)) for h,g in pulses]
                result=sequence_channel(steps,ideals,2,[np.kron(np.eye(2),g) for g in gates],mode='fresh')
                direct=channel_metrics(*result[:3],kraus=result[3],diamond=seed==900 and k in [2,32])
                endpoint_protocol=target_protocol(np.array([-1.,1.]),pulses,gates)
                score_contrast=float(np.linalg.norm(endpoint_protocol[0]-endpoint_protocol[1],2))
                for method,data,calls,resets in [('shared_numeric',numeric,4*2**m-2,0),('modular_gqsp_fresh',direct,k*plan.preparation_calls,k-1)]:
                    rows.append({'depth':depth,'seed':seed,'total_angle':total,'uses':k,'method':method,
                        'phase_bits':m if method=='shared_numeric' else '', 'gqsp_degree':plan.degree if method!='shared_numeric' else '',
                        'a_calls':calls,'resets':resets,'uniform_channel_certificate':2*total**2*sine_variance_bound(m) if method=='shared_numeric' else k*plan.channel_error_upper,
                        'ideal_score_endpoint_contrast':score_contrast,
                        'comparison_scope':'modular per-call GQSP, not global polynomial compilation',**data})
                assert numeric['choi_error']<=.01+1e-9 and direct['choi_error']<=.01+1e-9
        write(rows,'amortization.csv');print('amortization',depth,seed,len(rows),flush=True)


def amortization_resources():
    rows=[];total=.8;tolerance=.01;m=choose_phase_bits(total,tolerance);n=2**m
    table=-np.cos(2*np.pi*np.arange(n)/n)/np.cos(np.pi/n)
    for depth,input_bits in [(d,r) for d in [2,4,16] for r in [1,2]]:
        params=random_qru_params(depth,seed=900,scale=.6)
        step=1.6/(2**input_bits-1)
        e=affine_estimator(params,input_bits,-.7,step,AXIS,phase_bits=m,taper='sine',query_implementation='reflections')
        for k in [2,8,32]:
            gamma=total/k;plan=make_gqsp_plan(gamma,tolerance/k)
            target=e.num_qubits;numeric=QuantumCircuit(target+1)
            numeric.compose(e,qubits=range(target),inplace=True)
            for j in range(k):
                if j%2:numeric.h(target)
                numeric.append(UCRZGate((2*gamma*table).tolist()),[target,*range(input_bits,input_bits+m)])
                if j%2:numeric.h(target)
                if j<k-1:numeric.ry(.8/k,target)
            numeric.compose(e.inverse(),qubits=range(target),inplace=True)
            one,layout=gqsp_score_service(params,input_bits,-.7,step,AXIS,plan)
            direct=QuantumCircuit(one.num_qubits);dt=layout['target']
            for j in range(k):
                if j%2:direct.h(dt)
                direct.compose(one,inplace=True)
                if j%2:direct.h(dt)
                if j<k-1:
                    direct.ry(.8/k,dt)
                    direct.reset(layout['work'])
            for method,circuit,calls,episodes in [('shared_numeric',numeric,4*n-2,0),('modular_gqsp_fresh',direct,k*plan.preparation_calls,k-1)]:
                levels=[0,3] if k==32 and depth in [2,16] else [0]
                for level in levels:
                    compiled=transpile(circuit,basis_gates=['rz','sx','x','cx'],optimization_level=level,seed_transpiler=17)
                    rows.append({'depth':depth,'input_bits':input_bits,'uses':k,'method':method,'a_calls':calls,'reset_episodes':episodes,
                        'physical_single_qubit_resets':compiled.count_ops().get('reset',0),'qubits':circuit.num_qubits,
                        'cx':compiled.count_ops().get('cx',0),'compiled_depth':compiled.depth(),'optimization_level':level,
                        'total_angle':total,'tolerance':tolerance,'phase_bits':m if method=='shared_numeric' else '',
                        'gqsp_degree':plan.degree if method!='shared_numeric' else ''})
            write(rows,'amortization_resources.csv');print('amortization resources',depth,input_bits,k,flush=True)


def direct_memory(models=None,totals=None,uses=None,names=None,filename='direct_memory.csv',diamond_seed=900):
    rows=[];identity=np.eye(4);had=np.array([[1,1],[1,-1]])/np.sqrt(2)
    kick=np.kron(Z,np.eye(4));frame=np.kron(np.eye(4),had)
    for depth,seed in (SPEC['direct_memory']['models'] if models is None else models):
        params=random_qru_params(depth,seed=seed,scale=.6);xs=[-.7,.9]
        amps=[directional_amplitude_unitary(qru_unitary(params,x),AXIS) for x in xs]
        scores=np.array([signed_coordinate(qru_unitary(params,x)[:,0],AXIS) for x in xs])
        for total in ([.8,2.4] if totals is None else totals):
            for k in ([2,4,8,16,32] if uses is None else uses):
                gamma=total/k;plan=make_gqsp_plan(gamma,.01/k)
                zsteps=np.array([branch_dilation(a,plan) for a in amps])
                # Exact one-call invariance under an output-workspace unitary.
                base_ks=branch_kraus(zsteps,2);kicked_ks=branch_kraus(np.array([kick@w for w in zsteps]),2)
                one_call_residual=float(np.max(abs(kraus_choi(base_ks)-kraus_choi(kicked_ks))))
                assert one_call_residual<1e-12
                for name in (['repeated','alternating','echo','mixed'] if names is None else names):
                    hs=[X if name=='alternating' and j%2 else Z for j in range(k)]
                    steps=[np.array([frame@w@frame for w in zsteps]) if np.allclose(h,X) else zsteps for h in hs]
                    ideals=[block_diag(*generator_rotations(h,gamma,scores)) for h in hs]
                    if name in ['repeated','alternating']:interventions=[identity]*(k-1)
                    elif name=='echo':interventions=[np.kron(np.eye(2),X)]*(k-1)
                    else:interventions=[np.kron(had,X)]*(k-1)
                    local=[clean_contract(branch_kraus(w,2)[0],u) for w,u in zip(steps,ideals)]
                    for policy in ['fresh','shared','shared_phase_pi']:
                        actual=steps if policy!='shared_phase_pi' else [np.array([kick@w for w in step]) for step in steps]
                        result=sequence_channel(actual,ideals,2,interventions,mode='fresh' if policy=='fresh' else 'shared')
                        use_sdp=seed==diamond_seed and k==32 and (filename=='direct_memory.csv' or (total==.8 and name=='alternating'))
                        data=channel_metrics(*result[:3],kraus=result[3],diamond=use_sdp)
                        envelope=min(2.,sum(c.channel_bound for c in local)) if policy=='fresh' else shared_bound(local)
                        assert data['choi_error']<=envelope+1e-9
                        rows.append({'depth':depth,'seed':seed,'total_angle':total,'uses':k,'protocol':name,'policy':policy,
                            'a_calls':k*plan.preparation_calls,'reset_episodes':k-1 if policy=='fresh' else 0,
                            'local_contract_envelope':envelope,'single_call_channel_change':one_call_residual,
                            'extra_workspace_z_gates':k if policy=='shared_phase_pi' else 0,**data})
        write(rows,filename);print('direct memory',depth,seed,len(rows),flush=True)


def direct_memory_validation():
    spec=SPEC['direct_memory_validation']
    direct_memory(models=spec['models'],totals=spec['total_angles'],uses=spec['uses'],names=spec['protocols'],
                  filename='direct_memory_validation.csv',diamond_seed=1500)


STAGES={'precision':precision,'resources':resources,'fibre':fibre,'renewal':renewal,'inverse':inverse,'amortization':amortization,'amortization_resources':amortization_resources,'direct_memory':direct_memory,'direct_memory_validation':direct_memory_validation}


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all',*STAGES],default='all');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[*ROOT.joinpath('src').rglob('*.py'),Path(__file__)]
    config=SPEC|{'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'python':platform.python_version()}
    (OUT/f'protocol_{args.stage}.json').write_text(json.dumps(config,indent=2))
    for name in STAGES if args.stage=='all' else [args.stage]:
        start=time.perf_counter();STAGES[name]()
        print(name,'complete',round(time.perf_counter()-start,3),'seconds',flush=True)


if __name__=='__main__':main()
