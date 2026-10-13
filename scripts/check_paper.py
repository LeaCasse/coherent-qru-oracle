"""Compute paper summaries and compare complete reruns with archived tables."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
def read(folder,name,data=None):
 with ((data or ROOT/'reference_data')/folder/name).open() as f:return list(csv.DictReader(f))
def number(rows,key):return np.array([float(r[key]) for r in rows])
def summarize(data=None):
 direct=read('direct_control','channels.csv',data);cal=read('direct_control','calibrated.csv',data)
 process=read('process_study','channels.csv',data);gauge=read('process_study','gauge.csv',data)
 noise=read('process_study','gate_noise.csv',data)
 assert [len(direct),len(cal),len(process),len(gauge),len(noise)]==[432,72,648,24,24]
 precision={}
 for name in ['signed4','signed8','direct','calibrated']:
  r=[x for x in direct+cal if x['depth']=='2' and x['phase_bits']=='5' and x['window']=='sine' and x['policy']==name]
  precision[name]=float(number(r,'diamond_upper').mean())
 expected={'signed4':.07119,'signed8':.01306,'direct':.01070,'calibrated':.01038}
 for k,v in expected.items():assert abs(precision[k]-v)<5e-6,(k,precision[k],v)
 memory=[]
 for name in ['repeated_z','echo','alternating','input_mixed_echo']:
  groups=[]
  for mode in ['shared','fresh']:
   r=sorted([x for x in process if x['depth']=='2' and x['phase_bits']=='4' and x['repetitions']=='8' and x['protocol']==name and x['mode']==mode],key=lambda x:int(x['seed']))
   groups.append(number(r,'choi_trace_norm'))
  delta=groups[0]-groups[1];half=t.ppf(.975,5)*delta.std(ddof=1)/np.sqrt(6)
  memory.append({'protocol':name,'shared':float(groups[0].mean()),'fresh':float(groups[1].mean()),'paired_interval':(float(delta.mean()-half),float(delta.mean()+half))})
 witness=next(x for x in gauge if x['seed']=='603' and x['protocol']=='input_mixed_echo' and x['phase']=='2.1')
 assert abs(float(witness['diamond_upper'])-.30154)<5e-6
 assert abs(float(witness['sector_bound'])-.17134)<5e-6
 resources=read('direct_control','resources.csv',data)
 resource={x['policy']:{k:int(x[k]) for k in ['qubits','cx']} for x in resources if x['phase_bits']=='5' and x['window']=='sine'}
 assert resource['signed4']=={'qubits':15,'cx':11672}
 assert resource['multiplex4']=={'qubits':10,'cx':10726}
 noise_summary={kind:float(np.mean([float(x['noise_increment_trace_norm']) for x in noise if x['implementation']==kind and float(x['cx_depolarizing_parameter'])==1e-4])) for kind in ['unfused','fused']}
 return {'precision_m5_depth2':precision,'memory_m4_k8':memory,'fibre_counterexample':{k:float(witness[k]) for k in ['diamond_upper','sector_bound']},'resources_m5':resource,'noise_increment_p1e4':noise_summary}
def compare():
 """Keys, row order and all numerical columns checked. SDP tolerance is explicit."""
 results=[]
 for ref in sorted((ROOT/'reference_data').rglob('*.csv')):
  rel=ref.relative_to(ROOT/'reference_data');new=ROOT/'outputs/results'/rel
  assert new.exists(),f'Missing {new}. Run experiments first.'
  old=list(csv.DictReader(ref.open()));fresh=list(csv.DictReader(new.open()))
  assert len(old)==len(fresh),(rel,len(old),len(fresh))
  max_error=0.
  for i,(a,b) in enumerate(zip(old,fresh)):
   for k,v in a.items():
    assert k in b,(rel,k)
    try:
     av,bv=float(v),float(b[k])
     if np.isnan(av) and np.isnan(bv):continue
     error=abs(av-bv);max_error=max(max_error,error)
     tolerance=5e-5 if any(s in k for s in ['diamond','sdp','gauge']) else 1e-8
     assert np.isclose(av,bv,atol=tolerance,rtol=1e-7),(rel,i,k,av,bv)
    except ValueError:assert v==b[k],(rel,i,k,v,b[k])
  results.append({'table':str(rel),'rows':len(old),'max_numeric_difference':max_error})
 return results
def main():
 manifest=json.loads((ROOT/'reference_data/manifest.json').read_text())
 for name,digest in manifest['sha256'].items():
  assert hashlib.sha256((ROOT/'reference_data'/name).read_bytes()).hexdigest()==digest, name
 p=argparse.ArgumentParser();p.add_argument('--compare',action='store_true');args=p.parse_args()
 result=compare() if args.compare else summarize()
 out=ROOT/'outputs';out.mkdir(exist_ok=True)
 (out/('comparison.json' if args.compare else 'paper_values.json')).write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
