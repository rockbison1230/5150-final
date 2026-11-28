import csv
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

OUT = Path(__file__).resolve().parent / 'final_results'
CELLS = list(itertools.product(range(4), range(2), range(2)))
CELL_COUNT = len(CELLS)
SOLVER_OPTIONS = {'time_limit': 60, 'mip_rel_gap': 0}
SENSITIVITY = 6
FINE_AGES = [[0], [1], [2], [3]]
COARSE_AGES = [[0, 1], [2, 3]]
DATASET_SEEDS = [20261013, 20261023, 20261033, 20261043, 20261053]
DATASETS = {'independent': 100, 'correlated': 100, 'correlated_large': 1000}
EPSILONS = [0.1, 0.5, 1, 2, 5, 10]
DRAWS_PER_SEED = 40
THRESHOLDS = [3, 5, 10]


def generate_records(kind, n, seed):
    rng = np.random.default_rng(seed)
    age = rng.choice(4, n, p=[.35, .30, .20, .15])
    if kind == 'independent':
        sex = rng.choice(2, n, p=[.5, .5])
        population = rng.choice(2, n, p=[.9, .1])
    else:
        population = (rng.random(n) < np.array([.25, .12, .06, .03])[age]).astype(int)
        sex_one_share = np.array([[.40, .48, .55, .62], [.70, .55, .35, .25]])
        sex = (rng.random(n) < sex_one_share[population, age]).astype(int)
    return np.column_stack([age, sex, population])


def joint_counts(records):
    cell_index = {cell: i for i, cell in enumerate(CELLS)}
    return np.bincount([cell_index[tuple(r)] for r in records], minlength=CELL_COUNT)


def make_tables(age_groups):
    domains = [list(range(len(age_groups))), [0, 1], [0, 1]]
    grouped = [(next(i for i, g in enumerate(age_groups) if a in g), s, p) for a, s, p in CELLS]
    rows, labels = [], []
    for pair in [(0, 1), (0, 2), (1, 2)]:
        for levels in itertools.product(*(domains[i] for i in pair)):
            rows.append([int(all(c[i] == v for i, v in zip(pair, levels))) for c in grouped])
            labels.append((pair, levels))
    return np.array(rows, dtype=float), labels


class Variables:
    def __init__(self, **sizes):
        self.slices, start = {}, 0
        for name, size in sizes.items():
            self.slices[name] = slice(start, start + size)
            start += size
        self.total = start

    def __getitem__(self, name):
        return self.slices[name]

    def rows(self, count=1):
        return np.zeros((count, self.total))


def release_constraints(v, release, n):
    A = release['A']
    m = A.shape[0]
    if release['noisy'] is None:
        exact = v.rows(m); exact[:, v['x']] = A
        constraints = [LinearConstraint(exact, release['lower'], release['upper'])]
    else:
        above = v.rows(m); above[:, v['x']] = A; above[:, v['residual']] = -np.eye(m)
        below = v.rows(m); below[:, v['x']] = -A; below[:, v['residual']] = -np.eye(m)
        fit = v.rows(); fit[0, v['residual']] = 1
        constraints = [LinearConstraint(above, -np.inf, release['noisy']),
                       LinearConstraint(below, -np.inf, -release['noisy']),
                       LinearConstraint(fit, -np.inf, release['best_fit'] * (1 + 1e-7) + 1e-4)]
    total = v.rows(); total[0, v['x']] = 1
    return constraints + [LinearConstraint(total, n, n)]


def solve(v, objective, integer_parts, upper_bounds, constraints):
    integrality = np.zeros(v.total)
    for name in integer_parts:
        integrality[v[name]] = 1
    for options in (SOLVER_OPTIONS, {'time_limit': 60}, {'time_limit': 60, 'presolve': False}):
        result = milp(c=objective, integrality=integrality,
                      bounds=Bounds(np.zeros(v.total), upper_bounds),
                      constraints=constraints, options=options)
        if result.success:
            return result
    raise RuntimeError(result.message)


def best_l1_fit(A, noisy, n):
    m = A.shape[0]
    v = Variables(x=CELL_COUNT, residual=m)
    above = v.rows(m); above[:, v['x']] = A; above[:, v['residual']] = -np.eye(m)
    below = v.rows(m); below[:, v['x']] = -A; below[:, v['residual']] = -np.eye(m)
    total = v.rows(); total[0, v['x']] = 1
    objective = np.zeros(v.total); objective[v['residual']] = 1
    result = solve(v, objective, ['x'], np.r_[np.full(CELL_COUNT, n), np.full(m, np.inf)],
                   [LinearConstraint(above, -np.inf, noisy), LinearConstraint(below, -np.inf, -noisy),
                    LinearConstraint(total, n, n)])
    return float(result.fun)


def residual_count(release):
    return release['A'].shape[0] if release['noisy'] is not None else 0


def agreement(original, reconstruction, n):
    return float(np.minimum(original, reconstruction).sum()) / n


def agreement_range(release, original, n):
    r = residual_count(release)
    v = Variables(x=CELL_COUNT, overlap=CELL_COUNT, picks_original=CELL_COUNT, residual=r)
    base = release_constraints(v, release, n)
    identity = np.eye(CELL_COUNT)

    overlap_at_most_x = v.rows(CELL_COUNT)
    overlap_at_most_x[:, v['overlap']] = identity; overlap_at_most_x[:, v['x']] = -identity
    overlap_at_least_x = v.rows(CELL_COUNT)
    overlap_at_least_x[:, v['overlap']] = identity; overlap_at_least_x[:, v['x']] = -identity
    overlap_at_least_x[:, v['picks_original']] = n * identity
    overlap_at_least_original = v.rows(CELL_COUNT)
    overlap_at_least_original[:, v['overlap']] = identity
    overlap_at_least_original[:, v['picks_original']] = -n * identity

    upper_bounds = np.r_[np.full(CELL_COUNT, n), original.astype(float),
                         np.ones(CELL_COUNT), np.full(r, np.inf)]
    total_overlap = np.zeros(v.total); total_overlap[v['overlap']] = 1
    integer_parts = ['x', 'picks_original']
    worst = solve(v, total_overlap, integer_parts, upper_bounds,
                  base + [LinearConstraint(overlap_at_least_x, 0, np.inf),
                          LinearConstraint(overlap_at_least_original, original - n, np.inf)])
    best = solve(v, -total_overlap, integer_parts, upper_bounds,
                 base + [LinearConstraint(overlap_at_most_x, -np.inf, 0)])
    return worst.fun / n, -best.fun / n


def release_prior(estimates, age_groups, n):
    counts = np.clip(np.asarray(estimates, float), 0, None)
    g = len(age_groups)
    grouped_age = np.array([counts[2 * i] + counts[2 * i + 1] for i in range(g)])
    sex_by_population = counts[-4:]
    sex = np.array([sex_by_population[0] + sex_by_population[1], sex_by_population[2] + sex_by_population[3]])
    population = np.array([sex_by_population[0] + sex_by_population[2], sex_by_population[1] + sex_by_population[3]])
    age = np.zeros(4)
    for i, group in enumerate(age_groups):
        age[group] = grouped_age[i] / len(group)
    shares = [s / s.sum() if s.sum() > 0 else np.full(len(s), 1 / len(s)) for s in (age, sex, population)]
    return n * np.array([shares[0][a] * shares[1][s] * shares[2][p] for a, s, p in CELLS])


def informed_choice(release, n):
    r = residual_count(release)
    v = Variables(x=CELL_COUNT, distance=CELL_COUNT, residual=r)
    identity = np.eye(CELL_COUNT)
    above = v.rows(CELL_COUNT); above[:, v['distance']] = identity; above[:, v['x']] = -identity
    below = v.rows(CELL_COUNT); below[:, v['distance']] = identity; below[:, v['x']] = identity
    prior = release['prior']
    constraints = release_constraints(v, release, n) + [
        LinearConstraint(above, -prior, np.inf), LinearConstraint(below, prior, np.inf)]
    objective = np.zeros(v.total); objective[v['distance']] = 1
    result = solve(v, objective, ['x'],
                   np.r_[np.full(CELL_COUNT, n), np.full(CELL_COUNT + r, np.inf)], constraints)
    return np.rint(result.x[v['x']]).astype(int)


def round_to_total(values, n):
    rounded = np.floor(values).astype(int)
    largest_remainders = np.argsort(-(values - rounded))
    rounded[largest_remainders[:n - rounded.sum()]] += 1
    return rounded


def table_lines(age_group_count):
    g = age_group_count
    age_sex = np.arange(2 * g).reshape(g, 2)
    age_population = np.arange(2 * g, 4 * g).reshape(g, 2)
    sex_population = np.arange(4 * g, 4 * g + 4).reshape(2, 2)
    return [list(line) for table in (age_sex, age_population, sex_population)
            for line in itertools.chain(table, table.T)]


def secondary_suppression(counts, hidden, age_group_count):
    hidden = hidden.copy()
    changed = True
    while changed:
        changed = False
        for line in table_lines(age_group_count):
            line_hidden = [i for i in line if hidden[i]]
            line_visible = [i for i in line if not hidden[i]]
            if len(line_hidden) == 1 and line_visible:
                hidden[min(line_visible, key=lambda i: counts[i])] = True
                changed = True
    return hidden


def fine_query_estimates(published_estimates, age_groups):
    fine_A, fine_labels = make_tables(FINE_AGES)
    _, labels = make_tables(age_groups)
    position = {label: i for i, label in enumerate(labels)}
    group_of = {a: (i, len(g)) for i, g in enumerate(age_groups) for a in g}
    estimates = []
    for pair, levels in fine_labels:
        if 0 in pair:
            group, size = group_of[levels[0]]
            estimates.append(published_estimates[position[(pair, (group,) + tuple(levels[1:]))]] / size)
        else:
            estimates.append(published_estimates[position[(pair, levels)]])
    return np.array(estimates), fine_A


def deterministic_release(original, n, age_groups, threshold=None, secondary=False):
    A, _ = make_tables(age_groups)
    truth = A @ original
    hidden = np.zeros(len(truth), bool) if threshold is None else truth < threshold
    if secondary:
        hidden = secondary_suppression(truth, hidden, len(age_groups))
    hidden_ceiling = n if secondary else (threshold or 1) - 1
    lower = np.where(hidden, 0, truth)
    upper = np.where(hidden, hidden_ceiling, truth)
    published_estimates = np.where(hidden, ((threshold or 1) - 1) / 2, truth)
    return {'A': A, 'lower': lower, 'upper': upper, 'noisy': None, 'hidden': hidden,
            'published_estimates': published_estimates, 'age_groups': age_groups,
            'prior': release_prior(published_estimates, age_groups, n)}


def noisy_release(original, n, epsilon, rng):
    A, _ = make_tables(FINE_AGES)
    noisy = A @ original + rng.laplace(0, SENSITIVITY / epsilon, A.shape[0])
    return {'A': A, 'noisy': noisy, 'best_fit': best_l1_fit(A, noisy, n),
            'hidden': np.zeros(A.shape[0], bool), 'published_estimates': noisy,
            'age_groups': FINE_AGES, 'prior': release_prior(noisy, FINE_AGES, n)}


def suppressed_counts_inferable(release, n):
    hidden_rows = np.flatnonzero(release['hidden'])
    v = Variables(x=CELL_COUNT)
    constraints = release_constraints(v, release, n)
    inferable = 0
    for i in hidden_rows:
        objective = np.zeros(v.total); objective[v['x']] = release['A'][i]
        low = solve(v, objective, ['x'], np.full(CELL_COUNT, n), constraints).fun
        high = -solve(v, -objective, ['x'], np.full(CELL_COUNT, n), constraints).fun
        inferable += int(round(low) == round(high))
    return inferable, len(hidden_rows)


def evaluate(release, original, n):
    reconstruction = informed_choice(release, n)
    worst, best = agreement_range(release, original, n)
    estimates, fine_A = fine_query_estimates(release['published_estimates'], release['age_groups'])
    directly_published = len(release['age_groups']) == 4
    not_published = (release['hidden'].mean() if directly_published
                     else 1 - (4 - release['hidden'][-4:].sum()) / 20)
    return {'agreement': agreement(original, reconstruction, n),
            'agreement_min': worst, 'agreement_max': best,
            'prior_agreement': agreement(original, round_to_total(release['prior'], n), n),
            'query_mae': float(np.abs(estimates - fine_A @ original).mean()),
            'not_directly_published': float(not_published)}


def deterministic_settings():
    yield 'Unprotected', '', dict(age_groups=FINE_AGES)
    yield 'Coarsening', 'ages 4 to 2', dict(age_groups=COARSE_AGES)
    yield 'Coarsening', 'ages 4 to 1', dict(age_groups=[[0, 1, 2, 3]])
    for t in THRESHOLDS:
        yield 'Suppression', f'below {t}', dict(age_groups=FINE_AGES, threshold=t)
        yield 'Suppression + secondary', f'below {t}', dict(age_groups=FINE_AGES, threshold=t, secondary=True)
    yield 'Coarsening + suppression', 'ages 4 to 2, below 5', dict(age_groups=COARSE_AGES, threshold=5)


def run_dataset(job):
    dataset, seed = job
    n = DATASETS[dataset]
    records = generate_records('independent' if dataset == 'independent' else 'correlated', n, seed)
    original = joint_counts(records)
    rows = []
    for method, setting, options in deterministic_settings():
        release = deterministic_release(original, n, **options)
        row = {'dataset': dataset, 'n': n, 'seed': seed, 'method': method,
               'setting': setting, 'draw': '', **evaluate(release, original, n)}
        if release['hidden'].any():
            row['suppressed_inferable'], row['suppressed_total'] = suppressed_counts_inferable(release, n)
        rows.append(row)
    for epsilon in EPSILONS:
        rng = np.random.default_rng([seed, int(epsilon * 10)])
        for draw in range(DRAWS_PER_SEED):
            release = noisy_release(original, n, epsilon, rng)
            rows.append({'dataset': dataset, 'n': n, 'seed': seed, 'method': 'Laplace',
                         'setting': f'epsilon {epsilon}', 'draw': draw, **evaluate(release, original, n)})
    return rows


def summarize(rows):
    metrics = ['agreement', 'agreement_min', 'agreement_max', 'prior_agreement',
               'query_mae', 'not_directly_published']
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['method'], row['setting']), []).append(row)
    summary = []
    for (dataset, method, setting), members in groups.items():
        entry = {'dataset': dataset, 'method': method, 'setting': setting, 'runs': len(members)}
        for metric in metrics:
            values = np.array([m[metric] for m in members])
            entry[f'{metric}_mean'] = float(values.mean())
            entry[f'{metric}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            entry[f'{metric}_p2.5'] = float(np.percentile(values, 2.5))
            entry[f'{metric}_p97.5'] = float(np.percentile(values, 97.5))
        if any('suppressed_total' in m for m in members):
            entry['suppressed_inferable'] = sum(m.get('suppressed_inferable', 0) for m in members)
            entry['suppressed_total'] = sum(m.get('suppressed_total', 0) for m in members)
        summary.append(entry)
    return summary


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global DRAWS_PER_SEED
    if len(sys.argv) > 1:
        DRAWS_PER_SEED = int(sys.argv[1])
    OUT.mkdir(exist_ok=True)
    jobs = [(dataset, seed) for dataset in DATASETS for seed in DATASET_SEEDS]
    with Pool() as pool:
        rows = [row for result in pool.map(run_dataset, jobs) for row in result]
    write_csv(OUT / 'runs.csv', rows)
    summary = summarize(rows)
    write_csv(OUT / 'summary.csv', summary)
    (OUT / 'config.json').write_text(json.dumps(
        {'datasets': DATASETS, 'dataset_seeds': DATASET_SEEDS, 'epsilons': EPSILONS,
         'draws_per_seed': DRAWS_PER_SEED, 'thresholds': THRESHOLDS,
         'numpy_version': np.__version__, 'scipy_version': __import__('scipy').__version__}, indent=2))
    for entry in summary:
        print(f"{entry['dataset']:<17}{entry['method']:<26}{entry['setting']:<22}"
              f"agree {entry['agreement_mean']:.2f} [{entry['agreement_min_mean']:.2f}, "
              f"{entry['agreement_max_mean']:.2f}]  prior {entry['prior_agreement_mean']:.2f}  "
              f"mae {entry['query_mae_mean']:.2f}  ndp {entry['not_directly_published_mean']:.2f}"
              + (f"  inferable {entry['suppressed_inferable']}/{entry['suppressed_total']}"
                 if 'suppressed_total' in entry else ''))


if __name__ == '__main__':
    main()
