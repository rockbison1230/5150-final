import csv
import itertools
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

OUT = Path(__file__).resolve().parent
N = 100
DATA_SEED = 20261013
NOISE_SEED = 20261014
CELLS = list(itertools.product(range(4), range(2), range(2)))

def make_tables(age_groups):
    rows, labels = [], []
    domains = [list(range(len(age_groups))), [0, 1], [0, 1]]
    mapped = [(next(i for i,g in enumerate(age_groups) if a in g), s, p)
              for a,s,p in CELLS]
    for pair in [(0,1), (0,2), (1,2)]:
        for levels in itertools.product(*(domains[i] for i in pair)):
            rows.append([int(all(c[i] == v for i,v in zip(pair,levels)))
                         for c in mapped])
            labels.append({'attributes': list(pair), 'categories': list(levels)})
    return np.array(rows, dtype=float), labels

def solve_exact(A, lower, upper, objective=None):
    result = milp(c=np.zeros(16) if objective is None else objective,
                  integrality=np.ones(16), bounds=Bounds(0, N),
                  constraints=[LinearConstraint(A, lower, upper),
                               LinearConstraint(np.ones((1,16)), N, N)],
                  options={'time_limit': 15, 'mip_rel_gap': 0})
    if not result.success:
        raise RuntimeError(result.message)
    x = np.rint(result.x).astype(int)
    assert x.sum() == N and (x >= 0).all()
    assert (A @ x >= lower - 1e-6).all()
    assert (A @ x <= upper + 1e-6).all()
    return x

def exact_diagnostics(A, lower, upper):
    ranges = []
    for j in range(16):
        objective = np.eye(16)[j]
        lo = solve_exact(A, lower, upper, objective)[j]
        hi = solve_exact(A, lower, upper, -objective)[j]
        ranges.append([int(lo), int(hi)])
    return {'joint_cell_ranges': ranges,
            'uniquely_determined_joint_cells': sum(lo == hi for lo,hi in ranges),
            'unique_joint_histogram': all(lo == hi for lo,hi in ranges)}

def solve_noisy(A, y):
    m = len(y)
    c = np.r_[np.zeros(16), np.ones(m)]
    top = np.c_[A, -np.eye(m)]
    bottom = np.c_[-A, -np.eye(m)]
    fixed_n = np.r_[np.ones(16), np.zeros(m)][None,:]
    result = milp(c=c, integrality=np.r_[np.ones(16), np.zeros(m)],
                  bounds=Bounds(np.zeros(16+m), np.r_[np.full(16,N),np.full(m,np.inf)]),
                  constraints=[LinearConstraint(top, -np.inf, y),
                               LinearConstraint(bottom, -np.inf, -y),
                               LinearConstraint(fixed_n,N,N)],
                  options={'time_limit': 15, 'mip_rel_gap': 0})
    if not result.success:
        raise RuntimeError(result.message)
    x = np.rint(result.x[:16]).astype(int)
    assert x.sum() == N and (x >= 0).all()
    assert abs(np.abs(A @ x-y).sum()-result.fun) < 1e-5
    return x, float(result.fun)

def main():
    rng = np.random.default_rng(DATA_SEED)
    records = np.column_stack([rng.choice(4,N,p=[.35,.30,.20,.15]),
                               rng.choice(2,N,p=[.5,.5]),
                               rng.choice(2,N,p=[.9,.1])])
    lookup = {c:i for i,c in enumerate(CELLS)}
    original = np.bincount([lookup[tuple(r)] for r in records], minlength=16)
    A, labels = make_tables([[0],[1],[2],[3]])
    truth = A @ original
    sensitivity = int(max(np.abs(A[:,i]-A[:,j]).sum()
                          for i in range(16) for j in range(16)))
    assert sensitivity == 6
    assert np.linalg.matrix_rank(A) == 13
    runs = []
    releases = {}

    def report(name, reconstruction, mae, unavailable, diagnostic=None):
        overlap = int(np.minimum(original,reconstruction).sum())
        item = {'method':name, 'attribute_combination_agreement':overlap/N,
                'available_original_count_mae':float(mae),
                'unavailable_original_count_fraction':float(unavailable),
                'reconstructed_joint_counts':reconstruction.tolist()}
        if diagnostic is not None:
            item.update(diagnostic)
        runs.append(item)

    baseline = solve_exact(A, truth, truth)
    report('Unprotected', baseline, 0, 0, exact_diagnostics(A, truth, truth))
    releases['unprotected'] = {'labels':labels, 'counts':truth.tolist()}

    C, coarse_labels = make_tables([[0,1],[2,3]])
    coarse = C @ original
    cx = solve_exact(C, coarse, coarse)
    available = [i for i,row in enumerate(A) if any(np.array_equal(row,c) for c in C)]
    assert len(available) == 4
    report('Coarsening (4 age groups to 2)',cx,0,1-len(available)/len(A),
           exact_diagnostics(C,coarse,coarse))
    releases['coarsening'] = {'age_groups':[[0,1],[2,3]],
                              'labels':coarse_labels,'counts':coarse.tolist()}

    threshold = 5
    hidden = truth < threshold
    lower = np.where(hidden,0,truth)
    upper = np.where(hidden,threshold-1,truth)
    sx = solve_exact(A,lower,upper)
    diagnostic = exact_diagnostics(A,lower,upper)
    hidden_ranges = []
    for i in np.flatnonzero(hidden):
        lo = float(A[i] @ solve_exact(A,lower,upper,A[i]))
        hi = float(A[i] @ solve_exact(A,lower,upper,-A[i]))
        hidden_ranges.append({'query_index':int(i),'minimum':lo,'maximum':hi})
    diagnostic['suppressed_query_ranges'] = hidden_ranges
    diagnostic['uniquely_inferred_suppressed_counts'] = sum(
        r['minimum'] == r['maximum'] for r in hidden_ranges)
    report('Suppression (count below 5)',sx,0,hidden.mean(),diagnostic)
    releases['suppression'] = {'threshold':threshold,'labels':labels,
                              'counts':[None if h else int(v) for h,v in zip(hidden,truth)]}

    eps = 1
    noisy = truth + np.random.default_rng(NOISE_SEED).laplace(0,sensitivity/eps,len(truth))
    nx, residual = solve_noisy(A,noisy)
    report('Laplace (epsilon=1; one draw)',nx,np.abs(noisy-truth).mean(),0)
    runs[-1]['minimum_absolute_fit_residual'] = residual
    releases['laplace'] = {'epsilon':eps,'sensitivity':sensitivity,'noise_scale':6,
                           'labels':labels,'counts':noisy.tolist()}

    with (OUT/'original_records.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['age_group','sex_category','population_group'])
        writer.writerows(records.tolist())
    payload={'dataset_size':N,'data_seed':DATA_SEED,'noise_seed':NOISE_SEED,
             'generation_probabilities':{'age':[.35,.30,.20,.15],'sex':[.5,.5],
                                          'population_group':[.9,.1]},
             'numpy_version':np.__version__,'scipy_version':scipy.__version__,
             'original_joint_counts':original.tolist(),'original_query_counts':truth.tolist(),
             'query_matrix_rank':13,'results':runs}
    (OUT/'results.json').write_text(json.dumps(payload,indent=2))
    (OUT/'releases.json').write_text(json.dumps(releases,indent=2))
    with (OUT/'summary.csv').open('w',newline='') as f:
        fields=['method','attribute_combination_agreement','available_original_count_mae',
                'unavailable_original_count_fraction','uniquely_determined_joint_cells']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(runs)
    print(json.dumps({'results':runs,'original_joint_counts':original.tolist()},indent=2))

if __name__ == '__main__':
    main()
