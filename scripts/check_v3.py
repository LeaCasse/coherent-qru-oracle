"""Check the frozen V3 data and the numerical claims used in the manuscript."""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'reference_data/v3'


def read(name):
    with (DATA/name).open() as stream:
        return list(csv.DictReader(stream))


def mean(rows, key):
    return sum(float(row[key]) for row in rows)/len(rows)


def main():
    manifest = json.loads((DATA/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == expected, name
    counts = {'precision.csv':270, 'resources.csv':66, 'fibre.csv':648,
        'renewal_validation.csv':216, 'renewal_scaling.csv':64, 'inverse.csv':600,
        'amortization.csv':60, 'amortization_resources.csv':44,
        'direct_memory.csv':360, 'direct_memory_validation.csv':378,
        'noise/gate_noise.csv':24}
    for name, count in counts.items():
        assert len(read(name)) == count, name
    precision = read('precision.csv')
    single = {}
    for method, calls, expected in [('qpe_prepared',254,.0028057912453099357),
                                    ('gqsp',30,.002510842675586657)]:
        rows = [r for r in precision if r['method']==method and r['gamma']=='0.8'
                and r['tolerance']=='0.01' and r['contract']=='channel_only']
        assert len(rows)==9 and all(int(r['a_calls'])==calls for r in rows)
        value = mean(rows, 'diamond_upper')
        assert math.isclose(value, expected, abs_tol=5e-9)
        single[method] = value
    sensitivity = max(float(r['fibre_choi_sensitivity']) for r in read('fibre.csv')
                      if r['initialization']=='zero')
    assert sensitivity < 7e-14
    rows = [r for r in read('renewal_scaling.csv') if r['uses']=='64' and r['branches']=='16'
            and r['phase_bits']=='5' and r['target_dimension']=='4' and r['regime']=='fixed_total']
    assert len(rows)==1 and int(rows[0]['selected_segments'])==6
    assert math.isclose(float(rows[0]['selected_certificate']), .007115425932089641, abs_tol=1e-10)
    validation = {}
    for policy, expected in [('fresh',.0010081505461519487),('shared',.03479771519946037),
                             ('shared_phase_pi',.00004102044729720305)]:
        rows = [r for r in read('direct_memory_validation.csv') if r['policy']==policy
                and r['uses']=='32' and r['total_angle']=='0.8' and r['protocol']=='alternating']
        assert len(rows)==7 and all(float(r['single_call_channel_change'])==0 for r in rows)
        value = mean(rows, 'choi_error')
        assert math.isclose(value, expected, abs_tol=1e-10)
        validation[policy] = value
    output = ROOT/'outputs'
    output.mkdir(exist_ok=True)
    result = {'rows':counts, 'single_call_diamond_upper_mean':single,
              'new_model_choi_mean':validation, 'invariant_fibre_sensitivity':sensitivity}
    (output/'v3_values.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
