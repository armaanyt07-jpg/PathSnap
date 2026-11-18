"""Directed-network HMM with log-space Viterbi and explicit segment breaks."""
from dataclasses import dataclass, asdict
from heapq import heappop, heappush
from math import asin, cos, inf, isfinite, log, radians, sin, sqrt, exp


@dataclass(frozen=True)
class Config:
    beta_m: float = 50.0
    max_gap_s: float = 120.0
    max_speed_mps: float = 55.0
    distance_slack_m: float = 50.0

    def __post_init__(self):
        if any(not isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Decoder parameters must be finite and positive')


class Network:
    def __init__(self, arcs):
        self.arcs, self.adj = {}, {}
        for arc in arcs:
            aid, u, v, length = arc['arc_id'], arc['from_id'], arc['to_id'], arc['length_m']
            if aid in self.arcs or not isfinite(length) or length <= 0:
                raise ValueError('Duplicate arc or invalid length')
            self.arcs[aid] = (u, v, length)
            self.adj.setdefault(u, []).append((v, length, aid))

    def shortest(self, source, target, limit):
        queue, distances, parents = [(0.0, source)], {source: 0.0}, {}
        while queue:
            dist, node = heappop(queue)
            if dist != distances[node]:
                continue
            if node == target:
                path = []
                while node != source:
                    node, aid = parents[node]
                    path.append(aid)
                return dist, path[::-1]
            for nxt, length, aid in self.adj.get(node, []):
                score = dist + length
                if score <= limit and score < distances.get(nxt, inf):
                    distances[nxt], parents[nxt] = score, (node, aid)
                    heappush(queue, (score, nxt))
        return inf, []

    def travel(self, left, right, limit):
        a, b = left['arc_id'], right['arc_id']
        x, y = left['along_arc_m'], right['along_arc_m']
        if a == b and y >= x:
            return (y - x, [a]) if y - x <= limit else (inf, [])
        _, end, length = self.arcs[a]
        start, _, _ = self.arcs[b]
        remainder = length - x + y
        if remainder > limit:
            return inf, []
        distance, middle = self.shortest(end, start, limit - remainder)
        return distance + remainder, [a, *middle, b]


def gps_distance(a, b):
    lat1, lat2 = radians(a['lat']), radians(b['lat'])
    dlat, dlon = lat2 - lat1, radians(b['lon'] - a['lon'])
    value = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 6371000 * 2 * asin(min(1, sqrt(max(0, value))))


def decode(observations, arcs, config=None):
    cfg = config or Config()
    network = Network(arcs)
    obs = observations
    emissions, baseline = [], []
    previous_idx = previous_time = None
    for o in obs:
        for key in ('timestamp', 'lon', 'lat'):
            if not isfinite(o[key]):
                raise ValueError('Non-finite observation')
        if not -90 <= o['lat'] <= 90 or not -180 <= o['lon'] <= 180:
            raise ValueError('Invalid GPS coordinates')
        if previous_idx is not None and (o['obs_idx'] <= previous_idx or o['timestamp'] <= previous_time):
            raise ValueError('Observation indices and timestamps must strictly increase')
        previous_idx, previous_time = o['obs_idx'], o['timestamp']
        seen, weights = set(), []
        for c in o['candidates']:
            aid, along, p = c['arc_id'], c['along_arc_m'], c['probability']
            if aid in seen or aid not in network.arcs:
                raise ValueError('Duplicate or unknown candidate arc')
            if not isfinite(along) or not 0 <= along <= network.arcs[aid][2]:
                raise ValueError('Candidate projection outside arc')
            if not isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Invalid emission probability')
            seen.add(aid)
            weights.append(max(p, 1e-12))
        total = sum(weights)
        probabilities = [p / total for p in weights]
        emissions.append(probabilities)
        baseline.append(max(range(len(weights)), key=weights.__getitem__) if weights else None)

    chosen, breaks = [None] * len(obs), []
    start, scores, pointers = None, [], []

    def finish(end):
        if start is None:
            return
        state = max(range(len(scores)), key=scores.__getitem__)
        for t in range(end, start - 1, -1):
            chosen[t] = state
            if t > start:
                state = pointers[t - start][state]

    for t, o in enumerate(obs):
        reason = None
        if not o['candidates']:
            finish(t - 1)
            start, scores, pointers = None, [], []
            breaks.append({'obs_idx': o['obs_idx'], 'reason': 'missing_candidates'})
            continue
        e = [log(p) for p in emissions[t]]
        if start is not None:
            dt = o['timestamp'] - obs[t - 1]['timestamp']
            if dt > cfg.max_gap_s:
                reason = 'large_time_gap'
            else:
                limit = cfg.max_speed_mps * dt + cfg.distance_slack_m
                displacement = gps_distance(obs[t - 1], o)
                transitions = []
                for left in obs[t - 1]['candidates']:
                    row = []
                    for right in o['candidates']:
                        distance, _ = network.travel(left, right, limit)
                        row.append(-abs(distance - displacement) / cfg.beta_m)
                    finite = [v for v in row if isfinite(v)]
                    if finite:
                        m = max(finite)
                        norm = m + log(sum(exp(v - m) for v in finite))
                        row = [v - norm for v in row]
                    transitions.append(row)
                next_scores, back = [], []
                for j in range(len(e)):
                    alternatives = [scores[i] + row[j] for i, row in enumerate(transitions)]
                    best = max(range(len(alternatives)), key=alternatives.__getitem__)
                    next_scores.append(e[j] + alternatives[best])
                    back.append(best)
                if any(isfinite(s) for s in next_scores):
                    scores = next_scores
                    pointers.append(back)
                    continue
                reason = 'disconnected_or_speed_limit'
        if reason:
            finish(t - 1)
            breaks.append({'obs_idx': o['obs_idx'], 'reason': reason})
        start, scores, pointers = t, e, [None]
    finish(len(obs) - 1)

    break_ids = {b['obs_idx'] for b in breaks}
    output, segments, segment, baseline_route = [], [], [], []
    previous = None
    for t, o in enumerate(obs):
        c = o['candidates'][chosen[t]] if chosen[t] is not None else None
        b = o['candidates'][baseline[t]]['arc_id'] if baseline[t] is not None else None
        if b is None:
            if baseline_route and baseline_route[-1] is not None:
                baseline_route.append(None)
        elif not baseline_route or baseline_route[-1] != b:
            baseline_route.append(b)
        if c is None or o['obs_idx'] in break_ids:
            if segment:
                segments.append(segment)
            segment, previous = [], None
        if c is not None:
            if previous is None:
                segment.append(c['arc_id'])
            else:
                dt = o['timestamp'] - obs[t - 1]['timestamp']
                _, bridge = network.travel(previous, c, cfg.max_speed_mps * dt + cfg.distance_slack_m)
                segment.extend(bridge[1:])
            previous = c
        output.append({k: o[k] for k in ('obs_idx', 'timestamp', 'lon', 'lat')} | {
            'arc_id': c['arc_id'] if c else None, 'baseline_arc_id': b,
            'candidate_ids': [v['arc_id'] for v in o['candidates']],
            'probabilities': emissions[t]})
    if segment:
        segments.append(segment)
    route = []
    for segment in segments:
        if route:
            route.append(None)
        route.extend(segment)
    return {'observations': output, 'predicted_route': route, 'baseline_route': baseline_route,
            'route_segments': segments, 'breaks': breaks, 'decoder_config': asdict(cfg)}
