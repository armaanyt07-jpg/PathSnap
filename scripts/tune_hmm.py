"""Tune PathSnap's 20-feature HMM by observation accuracy on development trips."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from api.service import build_lattice
from hmm import Config, decode
from hmm.decoder import Network, gps_distance
from hmm.model import TrainedHMM
from pathsnap import evaluation, io, metrics, model_zoo
from pathsnap.config import artifacts_dir, load_config, results_dir
from pathsnap.features import HMM_FEATURE_COLUMNS, build_trip_features, centered_heading
from pathsnap.geo import LocalProjection, angle_diff_abs, bearing_deg


def prepare_trip(task):
    """Cache candidate features and separate reference labels for one trip."""
    split, trip_id, cfg, root = task
    path = root / f'{split}_{trip_id}.npz'
    if path.exists():
        return trip_id
    trip = io.load_trip(split, trip_id)
    frame, _ = build_trip_features(
        trip, k=int(cfg['k_candidates']), buffer_m=float(cfg['buffer_m']),
        max_segments=int(cfg['network_prefilter']['max_segments']),
        append_missing_labels=False, extended=True,
    )
    np.savez(
        path.with_suffix('.tmp.npz'),
        X=frame[HMM_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        obs_idx=frame['obs_idx'].to_numpy(dtype=np.int32),
        arc_id=frame['arc_id'].to_numpy(dtype=np.int32),
        along_m=frame['along_m'].to_numpy(dtype=np.float64),
        label_arc=frame['label_arc'].to_numpy(dtype=np.int32),
    )
    path.with_suffix('.tmp.npz').replace(path)
    return trip_id


def load_cache(root, split, trip_id, heading_window=1):
    with np.load(root / f'{split}_{trip_id}.npz') as cached:
        data = dict(cached)
    if heading_window != 1:
        trip = io.load_trip(split, trip_id, with_route=False)
        track, nodes, arcs = trip['track'], trip['nodes'], trip['arcs']
        projection = LocalProjection(track['lon'].median(), track['lat'].median())
        obs_xy = projection.xy(track['lon'], track['lat'])
        xy = projection.xy(nodes[:, 0], nodes[:, 1])
        segment_bearing = bearing_deg(xy[arcs[:, 0]], xy[arcs[:, 1]])
        headings = centered_heading(obs_xy, heading_window)
        data['X'][:, HMM_FEATURE_COLUMNS.index('heading_diff_deg')] = angle_diff_abs(
            headings[data['obs_idx']], segment_bearing[data['arc_id']]
        )
    return data


def fit_emissions(trips, seed, specs):
    """Fit candidate scorers using reference matches and sampled alternatives."""
    scores = []
    arc_length_col = HMM_FEATURE_COLUMNS.index('arc_len_m')
    for data in trips:
        valid = data['X'][:, arc_length_col] > 0
        y = (data['arc_id'] == data['label_arc']) & valid
        covered = np.isin(data['obs_idx'], data['obs_idx'][y])
        rng = np.random.default_rng(seed + len(scores))
        take = valid & covered & (y | (rng.random(len(y)) < 0.65))
        scores.append((data['X'][take], y[take]))
    X = np.concatenate([batch[0] for batch in scores])
    y = np.concatenate([batch[1] for batch in scores])
    scores.clear()
    fitted = []
    for name, params in specs:
        print(f'[emissions] {name}, {len(y):,} rows', flush=True)
        fitted.append((name, model_zoo.build('hist_gbdt', params, seed).fit(X, y)))
    return fitted


def lattice(root, split, trip_id, emitter, heading_window=1):
    data = load_cache(root, split, trip_id, heading_window)
    trip = io.load_trip(split, trip_id)
    track, nodes, arcs = trip['track'], trip['nodes'], trip['arcs']
    projection = LocalProjection(track['lon'].median(), track['lat'].median())
    xy = projection.xy(nodes[:, 0], nodes[:, 1])
    frame = pd.DataFrame({c: data[c] for c in ('obs_idx', 'arc_id', 'along_m')})
    obs, network, _ = build_lattice(track, arcs, xy, frame, emitter.predict_proba(data['X'])[:, 1])
    truth = np.full(len(obs), -1, dtype=int)
    truth[data['obs_idx']] = data['label_arc']
    return obs, network, truth, arcs


def decoder_grid():
    # Deterministic search over transition scale, emission strength, GPS jitter,
    # and the cost of an explicit route break.
    default = Config(beta_m=11.635646623114)
    grid = [default]
    for beta, odds, normalize, tolerance, restart, weight in product(
        (15, 50, 150), (False, True), (False, True), (0, 12, 35),
        (None, 2, 5, 10), (1, 2),
    ):
        grid.append(replace(default, beta_m=beta, emission_log_odds=odds,
                            normalize_transitions=normalize, backtrack_tolerance_m=tolerance,
                            restart_penalty=restart, emission_weight=weight))
    rng = np.random.default_rng(42)
    return [default, *[grid[i] for i in sorted(rng.choice(np.arange(1, len(grid)), 95, replace=False))]]


def refinement_grid():
    best = Config(beta_m=15, emission_weight=2, backtrack_tolerance_m=35, normalize_transitions=False)
    grid = [replace(best, beta_m=beta, emission_weight=weight, backtrack_tolerance_m=tolerance,
                    arc_switch_penalty=switch)
            for beta, weight, tolerance, switch in product(
                (6, 10, 15, 24, 36, 55), (0.75, 1.25, 2, 3.5), (12, 24, 35, 55, 80),
                (0, 0.15, 0.5, 1.0),
            )]
    rng = np.random.default_rng(43)
    return [best, *[grid[i] for i in sorted(rng.choice(len(grid), 143, replace=False)) if grid[i] != best]]


def transition_inputs(obs, network, k, cfg):
    """Measure feasible road travel once for the whole decoder search grid."""
    distance = np.full((len(obs), k, k), np.inf)
    backwards = distance.copy()
    displacement = np.zeros(len(obs))
    continuous_time = np.zeros(len(obs), dtype=bool)
    for t, (left, right) in enumerate(zip(obs, obs[1:]), 1):
        dt = right['timestamp'] - left['timestamp']
        if not 0 < dt <= cfg.max_gap_s:
            continue
        continuous_time[t] = True
        limit = cfg.max_speed_mps * dt + cfg.distance_slack_m
        displacement[t] = gps_distance(left, right)
        for i, source in enumerate(left['candidates']):
            for j, target in enumerate(right['candidates']):
                distance[t, i, j] = network.travel(source, target, limit)[0]
                step = source['along_arc_m'] - target['along_arc_m']
                if source['arc_id'] == target['arc_id'] and step > 0:
                    backwards[t, i, j] = step
    return distance, backwards, displacement, continuous_time


def search_trip(obs, network, truth, k, configs):
    """Run Viterbi for all configurations and count correct matches and breaks."""
    ids = np.full((len(obs), k), -1, dtype=int)
    p = np.zeros((len(obs), k))
    for t, o in enumerate(obs):
        for i, c in enumerate(o['candidates']):
            ids[t, i], p[t, i] = c['arc_id'], c['probability']
    available = ids >= 0
    log_p = np.where(available, np.log(np.maximum(p, 1e-12)), -np.inf)
    odds_bonus = -np.log(np.maximum(1 - p, 1e-12))
    beta = np.array([c.beta_m for c in configs])
    tol = np.array([c.backtrack_tolerance_m for c in configs])
    odds = np.array([c.emission_log_odds for c in configs])
    weights = np.array([c.emission_weight for c in configs])
    norm = np.array([c.normalize_transitions for c in configs])
    restart = np.array([c.restart_penalty or np.inf for c in configs])
    switch_penalty = np.array([c.arc_switch_penalty for c in configs])
    distance, backwards, displacement, continuous_time = transition_inputs(obs, Network(network), k, configs[0])
    pointers = np.zeros((len(obs), len(configs), k), dtype=np.int16)
    ends = np.zeros((len(obs), len(configs)), dtype=np.int16)
    setting = np.arange(len(configs))
    indices = np.arange(k)
    previous = np.full((len(configs), k), -np.inf)
    for t in range(len(obs)):
        emission = weights[:, None] * (log_p[t] + odds[:, None] * odds_bonus[t])
        pointer = np.full_like(pointers[t], -1)
        if continuous_time[t] and available[t - 1].any() and available[t].any():
            distance_t = np.where(backwards[t] <= tol[:, None, None], 0.0, distance[t])
            transition = -np.abs(distance_t - displacement[t]) / beta[:, None, None]
            changed_arc = ids[t - 1, :, None] != ids[t, None, :]
            transition -= switch_penalty[:, None, None] * changed_arc
            if norm.any():
                peak = np.max(transition, axis=2, keepdims=True)
                peak = np.where(np.isfinite(peak), peak, 0.0)
                total = np.exp(transition - peak).sum(axis=2, keepdims=True)
                offset = peak + np.log(np.where(total > 0, total, 1.0))
                transition -= np.where(norm[:, None, None], offset, 0.0)
            alternatives = previous[:, :, None] + transition
            best = np.argmax(alternatives, axis=1)
            best_score = alternatives[setting[:, None], best, indices]
            restart_score = np.max(previous, axis=1) - restart
            restart_mask = restart_score[:, None] > best_score
            best_score = np.maximum(best_score, restart_score[:, None])
            no_path = ~np.any(np.isfinite(best_score) & available[t], axis=1)
            best_score[no_path] = 0.0
            pointer = np.where(restart_mask | no_path[:, None], -1, best)
            emission += best_score
        pointers[t] = pointer
        ends[t] = np.argmax(emission, axis=1)
        previous = emission
    state = ends[-1]
    hits = np.zeros(len(configs), dtype=int)
    breaks = np.zeros(len(configs), dtype=int)
    predictions = np.empty((len(obs), len(configs)), dtype=int)
    for t in range(len(obs) - 1, -1, -1):
        predicted = ids[t, state]
        predictions[t] = predicted
        hits += (truth[t] >= 0) & (predicted == truth[t])
        parent = pointers[t, setting, state]
        if t:
            breaks += parent < 0
            state = np.where(parent < 0, ends[t - 1], parent)
    return hits, breaks, predictions


def validate(root, split, trip_ids, model):
    """Score the selected model with the production decoder and check connectivity."""
    rows = []
    for trip_id in trip_ids:
        obs, network, truth, arcs = lattice(root, split, trip_id, model, model.heading_window)
        decoded = decode(obs, network, model.decoder_config)
        for segment in decoded['route_segments']:
            if not np.array_equal(arcs[segment[:-1], 1], arcs[segment[1:], 0]):
                raise RuntimeError(f'Disconnected route segment for {trip_id}')
        row = {'trip_id': trip_id, 'observations': len(obs), 'breaks': len(decoded['breaks']),
               'candidate_coverage': float(np.mean([y in r['candidate_ids'] for y, r in zip(truth, decoded['observations'])]))}
        for key, field in (('hmm', 'arc_id'), ('independent', 'baseline_arc_id')):
            pred = np.array([r[field] if r[field] is not None else -1 for r in decoded['observations']])
            row[key] = metrics.classification_metrics(truth, pred)
            row[key]['route_continuity'] = metrics.route_continuity(pred, arcs)
        rows.append(row)
        print(f"[{split} eval] {trip_id} HMM={row['hmm']['top1_accuracy']:.4f}", flush=True)
    weights = [r['observations'] for r in rows]
    out = {'trips': len(rows), 'observations': sum(weights), 'breaks': sum(r['breaks'] for r in rows),
           'candidate_coverage': float(np.average([r['candidate_coverage'] for r in rows], weights=weights)),
           'per_trip': rows}
    for key in ('hmm', 'independent'):
        out[key] = {
            'top1_accuracy': float(np.average([r[key]['top1_accuracy'] for r in rows], weights=[r[key]['n'] for r in rows])),
            'route_continuity': float(np.average([r[key]['route_continuity'] for r in rows], weights=np.array(weights) - 1)),
        }
    return out


def tune(root, split, cfg, mode):
    """Select on development trips, refit, and evaluate the selected settings."""
    seed = int(cfg['seed'])
    train, test = cfg['splits']['train'], cfg['splits']['student_test']
    folds = evaluation.make_split(split['dev'], holdout_frac=0.25, seed=seed + 1)
    specs = [
        ('balanced', {}),
        ('unweighted', {'class_weight': None, 'max_leaf_nodes': 63, 'max_iter': 350, 'learning_rate': 0.07}),
        ('regularized', {'class_weight': None, 'max_leaf_nodes': 31, 'max_iter': 350, 'learning_rate': 0.05,
                         'l2_regularization': 5.0, 'min_samples_leaf': 100}),
    ]
    deep = {'max_leaf_nodes': 127, 'min_samples_leaf': 25, 'max_iter': 450, 'learning_rate': 0.06}
    if mode == 'refined':
        specs = [('balanced', {}), ('deep', deep)]
    elif mode == 'heading':
        specs = [('deep', deep), ('heading_3', deep), ('heading_6', deep)]
    windows = [1, 3, 6] if mode == 'heading' else [1] * len(specs)
    if mode == 'heading':
        estimated = [fit_emissions((load_cache(root, train, t, window) for t in folds['dev']),
                                   seed, [spec])[0] for window, spec in zip(windows, specs)]
    else:
        estimated = fit_emissions((load_cache(root, train, t) for t in folds['dev']), seed, specs)
    votes = np.zeros(len(estimated), dtype=int)
    configs = decoder_grid() if mode == 'tuned' else refinement_grid()
    if mode == 'heading':
        configs = [Config(beta_m=10, emission_weight=1.25, backtrack_tolerance_m=55,
                          normalize_transitions=False, arc_switch_penalty=0.5), *configs[:47]]
    hits = np.zeros((len(estimated), len(configs)), dtype=int)
    breaks = np.zeros_like(hits)
    labeled = 0
    for i, trip_id in enumerate(folds['holdout']):
        for j, (_, emitter) in enumerate(estimated):
            obs, network, truth, _ = lattice(root, train, trip_id, emitter, windows[j])
            if j == 0:
                labeled += int((truth >= 0).sum())
            trip_hits, trip_breaks, predictions = search_trip(obs, network, truth, int(cfg['k_candidates']), configs)
            if i == 0:
                for candidate in (0, 1, 2, 3):
                    rows = decode(obs, network, configs[candidate])['observations']
                    np.testing.assert_array_equal(predictions[:, candidate], [r['arc_id'] if r['arc_id'] is not None else -1 for r in rows])
            hits[j] += trip_hits
            breaks[j] += trip_breaks
            votes[j] += sum(bool(o['candidates']) and max(o['candidates'], key=lambda c: c['probability'])['arc_id'] == y
                            for o, y in zip(obs, truth))
        print(f'[tune] {i + 1}/{len(folds["holdout"])} {trip_id}, best accuracy {hits.max() / labeled:.4f}', flush=True)

    j, d = max(np.ndindex(hits.shape), key=lambda index: (hits[index], -breaks[index]))
    decoder = configs[d]
    print(f'[selected] {specs[j][0]} validation={hits[j,d] / labeled:.4f}, {decoder}', flush=True)
    window = windows[j]
    final_estimator = fit_emissions((load_cache(root, train, t, window) for t in split['dev']), seed, [specs[j]])[0][1]
    model = TrainedHMM(final_estimator, decoder, heading_window=window)
    name = f'hmm_20_{mode}'
    bundle = artifacts_dir() / f'{name}.joblib'
    joblib.dump(model, bundle)
    model = joblib.load(bundle)
    report = {
        'trained_utc': evaluation.utc_now(), 'features': list(model.feature_columns),
        'decoder_config': asdict(decoder),
        'emission_params': model_zoo.DEFAULT_PARAMS['hist_gbdt'] | specs[j][1],
        'heading_window': window,
        'trial_count': int(hits.size), 'validation_split': folds,
        'validation_top1_accuracy': float(hits[j,d] / labeled),
        'validation_independent_accuracy': float(votes[j] / labeled),
        'holdout': validate(root, train, split['holdout'], model),
        'student_test': validate(root, test, io.trip_ids(test), model),
    }
    path = results_dir() / f'{name}_metrics.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {bundle} and {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--refine', dest='mode', action='store_const', const='refined',
                       help='Search around the first tuning run’s best decoder')
    modes.add_argument('--heading-search', dest='mode', action='store_const', const='heading',
                       help='Validate wider GPS heading windows with the same 20 model inputs')
    parser.set_defaults(mode='tuned')
    args = parser.parse_args()
    cfg = load_config()
    train, test = cfg['splits']['train'], cfg['splits']['student_test']
    split = evaluation.load_or_make_split(artifacts_dir() / 'split.json', io.trip_ids(train), seed=int(cfg['seed']))
    cache = artifacts_dir() / 'hmm_tuning_cache_v1'
    cache.mkdir(exist_ok=True)
    tasks = [(train, trip_id, cfg, cache) for trip_id in split['dev'] + split['holdout']]
    tasks.extend((test, trip_id, cfg, cache) for trip_id in io.trip_ids(test))
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        for i, trip_id in enumerate(workers.map(prepare_trip, tasks), 1):
            print(f'[prepare] {i}/{len(tasks)} {trip_id}', flush=True)
    if not args.prepare_only:
        tune(cache, split, cfg, args.mode)


if __name__ == '__main__':
    main()
