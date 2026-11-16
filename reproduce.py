"""Reproduce saved figures, run experiments or verify the paper."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LEGACY_STUDIES = ['channel_study', 'precision_study', 'direct_control_study',
    'calibrated_control_study', 'process_study', 'gate_noise_pilot',
    'reuse_control_study', 'polynomial_baseline_audit', 'renewal_study']
V3_STUDIES = ['precision', 'resources', 'fibre', 'renewal', 'inverse',
    'amortization', 'amortization_resources', 'direct_memory',
    'direct_memory_validation', 'noise']
LEGACY_FIGURES = ['build_figures', 'build_appendix_channels', 'build_bit_allocation',
    'build_quaternion_fibre', 'build_overview', 'build_paper_circuits', 'build_renewal']
PAPER_FIGURES = ['06_renewal', '02_precision',
    'A_03_scalar_vs_channel', 'A_07_quaternion_fibre', 'A_02_fibre_cancellation',
    '04_fibre', '03_memory', 'A_04_dephasing_intervention']
NEW_FIGURES = ['v3_memory', 'v3_design', 'v3_geometry', 'v3_inverse', 'v3_precision', 'v3_bits',
    'v3_framework', 'v3_circuits', 'v3_interface']


def command(path, *args, extra_env=None):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + env.get('PYTHONPATH', '')
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = '1'
    env.update(extra_env or {})
    subprocess.run([sys.executable, str(path), *args], cwd=ROOT, env=env, check=True)


def figures(version, data):
    output = ROOT/'outputs/figures'
    output.mkdir(parents=True, exist_ok=True)
    legacy_data = data if version == 'v2' and data else ROOT/'reference_data'
    for name in LEGACY_FIGURES:
        command(ROOT/'figures'/f'{name}.py', extra_env={'QRU_DATA_DIR': str(legacy_data.resolve())})
    if version == 'v2':
        return
    new_data = data or ROOT/'reference_data/v3'
    command(ROOT/'figures/build_v3.py', extra_env={'QRU_V3_DATA_DIR': str(new_data.resolve())})
    command(ROOT/'figures/build_v3_schematics.py')
    paper = output/'paper'
    paper.mkdir(exist_ok=True)
    for stale in paper.glob('*.pdf'):
        if stale.stem not in PAPER_FIGURES + NEW_FIGURES:
            stale.unlink()
    for name in PAPER_FIGURES + NEW_FIGURES:
        source = output/('v3' if name in NEW_FIGURES else '')/(name+'.pdf')
        assert source.read_bytes().rstrip().endswith(b'%%EOF'), source
        shutil.copyfile(source, paper/source.name)
    print(f'Paper figures: {paper} ({len(PAPER_FIGURES) + len(NEW_FIGURES)} PDFs)')


def experiments(version, study):
    allowed = LEGACY_STUDIES if version == 'v2' else V3_STUDIES
    if study != 'all' and study not in allowed:
        raise SystemExit(f'Unknown {version} study: {study}. Choose from {allowed}')
    if version == 'v2':
        for name in allowed if study == 'all' else [study]:
            command(ROOT/'experiments'/f'{name}.py')
        return
    if study != 'noise':
        command(ROOT/'experiments/paper_study.py', '--stage', study)
    if study in ['all', 'noise']:
        command(ROOT/'experiments/gate_noise_pilot.py', '--optimized')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode')
    plot = sub.add_parser('figures', help='Build manuscript figures from saved tables')
    plot.add_argument('--version', choices=['v2', 'v3'], default='v3')
    plot.add_argument('--data', type=Path)
    run = sub.add_parser('experiments', help='Recompute numerical studies')
    run.add_argument('--version', choices=['v2', 'v3'], default='v3')
    run.add_argument('--study', default='all')
    sub.add_parser('verify', help='Check saved data, circuit identities and tests')
    sub.add_parser('compare', help='Compare a complete V2 rerun with its saved tables')
    sub.add_parser('notebook', help='Execute the earlier worked examples')
    revision = sub.add_parser('revision', help=argparse.SUPPRESS)
    revision.add_argument('--stage', choices=['all', *V3_STUDIES], default='all')
    args = parser.parse_args()
    mode = args.mode or 'figures'
    if mode == 'figures':
        figures(getattr(args, 'version', 'v3'), getattr(args, 'data', None))
    elif mode in ['experiments', 'revision']:
        experiments(getattr(args, 'version', 'v3'), getattr(args, 'study', getattr(args, 'stage', 'all')))
    elif mode == 'verify':
        command(ROOT/'scripts/check_paper.py')
        command(ROOT/'scripts/check_v3.py')
        command(ROOT/'figures/verify_circuit_structure.py')
        command('-m', 'pytest', 'tests')
    elif mode == 'compare':
        command(ROOT/'scripts/check_paper.py', '--compare')
    else:
        command(ROOT/'scripts/execute_notebook.py')


if __name__ == '__main__':
    main()
