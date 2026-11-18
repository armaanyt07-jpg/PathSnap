"""C-owned integration: reference-free inference and server-only scoring."""
from time import perf_counter

import joblib
import numpy as np

from hmm import Config, decode
from pathsnap import baseline, features, labels, metrics, io
from pathsnap.geo import LocalProjection


def finite_metrics(values):
    return {k: v for k, v in values.items() if not isinstance(v, float) or np.isfinite(v)}


def predict_trip(trip, model, cfg, decoder_config=None):
    start = perf_counter()
    frame, _ = features.build_trip_features(dict(trip, route=None),
        k=int(cfg['k_candidates']), buffer_m=float(cfg['buffer_m']),
        max_segments=int(cfg['network_prefilter']['max_segments']))
    feature_end = perf_counter()
    scores = baseline.predict_scores(model, frame) if len(frame) else np.array([])
    if scores.shape != (len(frame),):
        raise ValueError('Expected one model score per feature row')
    track, nodes, arcs = trip['track'], trip['nodes'], trip['arcs']
    projection = LocalProjection(track['lon'].median(), track['lat'].median())
    xy = projection.xy(nodes[:, 0], nodes[:, 1])
    lengths = np.linalg.norm(xy[arcs[:, 1]] - xy[arcs[:, 0]], axis=1)
    network = [{'arc_id': i, 'from_id': int(a[0]), 'to_id': int(a[1]), 'length_m': float(lengths[i])}
               for i, a in enumerate(arcs) if lengths[i] > 0]
    candidates = [[] for _ in range(len(track))]
    for row, probability in zip(frame.itertuples(index=False), scores):
        if lengths[row.arc_id] > 0:
            candidates[row.obs_idx].append({'arc_id': int(row.arc_id),
                'along_arc_m': float(row.along_m), 'probability': float(probability)})
    observations = [{'obs_idx': i, 'timestamp': float(row.t), 'lon': float(row.lon),
                     'lat': float(row.lat), 'candidates': candidates[i]}
                    for i, row in enumerate(track.itertuples(index=False))]
    model_end = perf_counter()
    decoded = decode(observations, network, decoder_config or Config())
    decode_end = perf_counter()
    hmm_ids = np.array([o['arc_id'] if o['arc_id'] is not None else -1 for o in decoded['observations']], dtype=int)
    base_ids = np.array([o['baseline_arc_id'] if o['baseline_arc_id'] is not None else -1 for o in decoded['observations']], dtype=int)
    def continuity(ids):
        if len(ids) < 2:
            return {}
        good = (ids[:-1] >= 0) & (ids[1:] >= 0)
        connected = np.zeros(len(ids) - 1, dtype=bool)
        a, b = ids[:-1][good], ids[1:][good]
        connected[good] = (a == b) | (arcs[a, 1] == arcs[b, 0])
        return {'route_continuity': float(connected.mean())}
    hmm_metrics, base_metrics = continuity(hmm_ids), continuity(base_ids)
    route = trip.get('route')
    if route is not None and len(route):
        obs_xy = projection.xy(track['lon'], track['lat'])
        truth = labels.map_observations_to_route(obs_xy, route, xy[arcs[:, 0]], xy[arcs[:, 1]])['label_arc']
        hmm_metrics.update(finite_metrics(metrics.classification_metrics(truth, hmm_ids)))
        base_metrics.update(finite_metrics(metrics.classification_metrics(truth, base_ids)))
        valid_obs = truth >= 0
        hits = np.array([truth[i] in [c['arc_id'] for c in cs] for i, cs in enumerate(candidates)])
        common = {}
        if valid_obs.any():
            common['candidate_miss_rate'] = float(1 - hits[valid_obs].mean())
        if len(frame):
            row_truth = truth[frame['obs_idx'].to_numpy(int)]
            valid_rows = row_truth >= 0
            binary = frame['arc_id'].to_numpy()[valid_rows] == row_truth[valid_rows]
            common.update(finite_metrics(metrics.candidate_prf(binary, scores[valid_rows])))
            common.update(finite_metrics({'candidate_log_loss': metrics.candidate_log_loss(binary, scores[valid_rows])}))
        hmm_metrics.update(common)
        base_metrics.update(common)
    lines = []
    for segment in decoded['route_segments']:
        points = [nodes[arcs[segment[0], 0]].tolist()]
        points.extend(nodes[arcs[aid, 1]].tolist() for aid in segment)
        lines.append(points)
    return {'trip_id': trip['trip_id'], 'decoder': 'hmm-viterbi', 'n_observations': len(track),
        'observations': track[['lon', 'lat']].round(7).values.tolist(),
        'predictions': decoded['observations'],
        'predicted_route': [a for segment in decoded['route_segments'] for a in segment],
        'route_segments': decoded['route_segments'],
        'predicted_line': max(lines, key=len, default=[]), 'predicted_lines': lines,
        'baseline_route': [int(a) for a in base_ids if a >= 0],
        'breaks': decoded['breaks'], 'decoder_config': decoded['decoder_config'],
        'metrics': hmm_metrics, 'baseline_metrics': base_metrics,
        'metric_definitions': {'macro_f1': 'D helper: union of reference and predicted arc classes',
            'route_continuity': 'adjacent observation predictions: same arc or directly connected; missing counts as disconnected',
            'candidate_f1': 'binary scores thresholded at 0.5; shared by baseline and HMM'},
        'timings': {'features_ms': (feature_end-start)*1000, 'model_ms': (model_end-feature_end)*1000,
            'decode_ms': (decode_end-model_end)*1000, 'inference_ms': (decode_end-feature_end)*1000,
            'total_ms': (perf_counter()-start)*1000}}


class Service:
    """Serves the promoted bundle plus optional per-family comparison models."""

    def __init__(self, model, cfg, extra_models=None):
        self.model, self.cfg = model, cfg
        self._extra = dict(extra_models or {})
        self._loaded = {}
        self.default_name = 'champion'

    def model_names(self):
        return [self.default_name, *sorted(self._extra)]

    def resolve_model(self, name=None):
        if not name or name == self.default_name:
            return self.model
        if name not in self._extra:
            raise KeyError(name)
        if name not in self._loaded:
            self._loaded[name] = joblib.load(self._extra[name])
        return self._loaded[name]

    def trip_ids(self):
        return io.trip_ids(self.cfg['splits']['student_test'])

    def predict(self, trip_id, model_name=None):
        return predict_trip(io.load_trip(self.cfg['splits']['student_test'], trip_id),
                            self.resolve_model(model_name), self.cfg)
