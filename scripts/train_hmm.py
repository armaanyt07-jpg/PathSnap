"""Fit and evaluate the 20-feature PathSnap HMM on the existing dataset."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from api.service import build_lattice, predict_trip
from hmm import Config
from hmm.decoder import Network, gps_distance
from hmm.model import TrainedHMM
from pathsnap import evaluation, io, model_zoo
from pathsnap.config import artifacts_dir, load_config, results_dir
from pathsnap.features import HMM_FEATURE_COLUMNS, build_trip_features
from pathsnap.geo import LocalProjection


def transition_examples(trip, frame, cfg, k):
    """Collect Viterbi transition residuals for consecutive reference matches."""
    track, nodes = trip['track'], trip['nodes']
    projection = LocalProjection(track['lon'].median(), track['lat'].median())
    xy = projection.xy(nodes[:, 0], nodes[:, 1])
    obs, arcs, _ = build_lattice(track, trip['arcs'], xy, frame, np.zeros(len(frame)))
    network = Network(arcs)
    truth = frame.drop_duplicates('obs_idx').set_index('obs_idx')['label_arc'].to_dict()
    errors, targets = [], []
    for left, right in zip(obs, obs[1:]):
        dt = right['timestamp'] - left['timestamp']
        if not 0 < dt <= cfg.max_gap_s:
            continue
        source = next((c for c in left['candidates'] if c['arc_id'] == truth.get(left['obs_idx'])), None)
        target = next((j for j, c in enumerate(right['candidates'])
                       if c['arc_id'] == truth.get(right['obs_idx'])), None)
        if source is None or target is None:
            continue
        limit = cfg.max_speed_mps * dt + cfg.distance_slack_m
        displacement = gps_distance(left, right)
        residuals = [abs(network.travel(source, c, limit)[0] - displacement)
                     for c in right['candidates']]
        if np.isfinite(residuals[target]):
            errors.append(residuals + [np.inf] * (k - len(residuals)))
            targets.append(residuals[target])
    return np.asarray(errors).reshape(-1, k), np.asarray(targets)


def fit_transitions(examples, cfg):
    errors = np.concatenate([example[0] for example in examples])
    targets = np.concatenate([example[1] for example in examples])
    if not len(targets):
        raise ValueError('No feasible labeled transitions in the development trips')

    def loss(log_beta):
        beta = np.exp(log_beta)
        return float(np.mean(targets / beta + logsumexp(-errors / beta, axis=1)))

    fit = minimize_scalar(loss, bounds=(np.log(0.5), np.log(5000)), method='bounded')
    if not fit.success:
        raise RuntimeError(f'Transition fitting failed: {fit.message}')
    return replace(cfg, beta_m=float(np.exp(fit.x))), len(targets)


def evaluate(split, trip_ids, model, cfg):
    rows = []
    for trip_id in trip_ids:
        trip = io.load_trip(split, trip_id)
        result = predict_trip(trip, model, cfg)
        arcs = trip['arcs']
        for segment in result['route_segments']:
            if not np.array_equal(arcs[segment[:-1], 1], arcs[segment[1:], 0]):
                raise RuntimeError(f'Disconnected reconstructed segment in {trip_id}')
        rows.append({
            'trip_id': trip_id,
            'observations': result['n_observations'],
            'breaks': len(result['breaks']),
            'hmm': result['metrics'],
            'independent': result['baseline_metrics'],
        })
        print(f"[{split} eval] {trip_id} HMM top-1={result['metrics']['top1_accuracy']:.4f}", flush=True)

    summary = {'trips': len(rows), 'observations': sum(r['observations'] for r in rows),
               'breaks': sum(r['breaks'] for r in rows), 'per_trip': rows}
    for method in ('hmm', 'independent'):
        summary[method] = {
            'top1_accuracy': float(np.average([r[method]['top1_accuracy'] for r in rows],
                                              weights=[r[method]['n'] for r in rows])),
            'mean_trip_macro_f1': float(np.mean([r[method]['macro_f1'] for r in rows])),
            'route_continuity': float(np.average([r[method]['route_continuity'] for r in rows],
                                                weights=[r['observations'] - 1 for r in rows])),
        }
    return summary


def main():
    cfg = load_config()
    seed, k = int(cfg['seed']), int(cfg['k_candidates'])
    train, test = cfg['splits']['train'], cfg['splits']['student_test']
    split = evaluation.load_or_make_split(artifacts_dir() / 'split.json', io.trip_ids(train), seed=seed)
    decoder_cfg = Config()
    samples, transitions = [], []
    for i, trip_id in enumerate(split['dev'], 1):
        trip = io.load_trip(train, trip_id)
        frame, _ = build_trip_features(
            trip, k=k, buffer_m=float(cfg['buffer_m']),
            max_segments=int(cfg['network_prefilter']['max_segments']),
            append_missing_labels=False, extended=True,
        )
        frame = frame[frame['arc_len_m'] > 0]
        transitions.append(transition_examples(trip, frame, decoder_cfg, k))
        labeled = evaluation.lattice_view(frame, k)
        training = evaluation.subsample_negatives(labeled, 0.5, seed + i)
        samples.append((training[HMM_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
                        training['label'].to_numpy(dtype=np.int8)))
        print(f"[fit features] {i}/{len(split['dev'])} {trip_id} rows={len(training):,}", flush=True)

    X = np.concatenate([sample[0] for sample in samples])
    y = np.concatenate([sample[1] for sample in samples])
    samples.clear()
    emission = model_zoo.build('hist_gbdt', seed=seed).fit(X, y)
    decoder_cfg, transition_count = fit_transitions(transitions, decoder_cfg)
    model = TrainedHMM(emission, decoder_cfg)
    bundle = artifacts_dir() / 'model_bundle.joblib'
    joblib.dump(model, bundle)
    model = joblib.load(bundle)
    print(f'Fitted {X.shape[1]} features on {len(X):,} candidates. beta={decoder_cfg.beta_m:.3f} m', flush=True)

    report = {
        'trained_utc': evaluation.utc_now(),
        'dataset': 'PathSnap train and student_test',
        'split': split,
        'artifact': str(bundle.relative_to(artifacts_dir().parent)),
        'emission_family': 'hist_gbdt',
        'features': list(model.feature_columns),
        'training_candidate_rows': len(X),
        'training_transition_pairs': transition_count,
        'decoder_config': asdict(decoder_cfg),
        'holdout': evaluate(train, split['holdout'], model, cfg),
        'student_test': evaluate(test, io.trip_ids(test), model, cfg),
    }
    path = results_dir() / 'hmm_20_metrics.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {bundle} and {path}', flush=True)


if __name__ == '__main__':
    main()
