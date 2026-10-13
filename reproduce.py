"""One entry point for figures, numerical campaigns, verification and notebook."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
STUDIES=['channel_study','precision_study','direct_control_study','calibrated_control_study','process_study','gate_noise_pilot','reuse_control_study','polynomial_baseline_audit']
FIGURES=['build_figures','build_appendix_channels','build_bit_allocation','build_quaternion_fibre','build_overview','build_paper_circuits']
def command(path,*args):
 env=os.environ.copy()
 env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
 subprocess.run([sys.executable,str(path),*args],cwd=ROOT,env=env,check=True)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 sub=parser.add_subparsers(dest='mode')
 f=sub.add_parser('figures',help='Render all 20 figures from reference or recomputed CSVs')
 f.add_argument('--data',type=Path,default=ROOT/'reference_data')
 e=sub.add_parser('experiments',help='Recompute numerical studies into outputs/results')
 e.add_argument('--study',choices=['all']+STUDIES,default='all')
 sub.add_parser('verify',help='Run channel/circuit tests and verify paper values')
 sub.add_parser('notebook',help='Execute the supplied notebook and export HTML')
 c=sub.add_parser('compare',help='Compare recomputed results to reference data')
 args=parser.parse_args();mode=args.mode or 'figures'
 (ROOT/'outputs/figures').mkdir(parents=True,exist_ok=True)
 if mode=='figures':
  os.environ['QRU_DATA_DIR']=str(getattr(args,'data',ROOT/'reference_data').resolve())
  for name in FIGURES:command(ROOT/'figures'/f'{name}.py')
  pdfs=[p for p in (ROOT/'outputs/figures').glob('*.pdf') if not p.name.startswith('_tmp_')]
  assert len(pdfs)==20
  for p in pdfs:assert p.read_bytes().rstrip().endswith(b'%%EOF'),f'Incomplete PDF: {p}'
 elif mode=='experiments':
  (ROOT/'outputs/results').mkdir(parents=True,exist_ok=True)
  for name in STUDIES if args.study=='all' else [args.study]:
   print(f'Running {name}',flush=True);command(ROOT/'experiments'/f'{name}.py')
 elif mode=='verify':
  command(ROOT/'scripts/check_paper.py');command(ROOT/'figures/verify_circuit_structure.py')
  subprocess.run([sys.executable,'-m','pytest','tests'],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},check=True)
 elif mode=='compare':command(ROOT/'scripts/check_paper.py','--compare')
 else:command(ROOT/'scripts/execute_notebook.py')
if __name__=='__main__':main()
