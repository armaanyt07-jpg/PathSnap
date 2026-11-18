"""Run from repo root: python scripts/run_decode.py --demo."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hmm import Config, decode


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'student_test'])
    parser.add_argument('--trip', action='append')
    parser.add_argument('--model', type=Path, default=root / 'artifacts/model_bundle.joblib')
    parser.add_argument('--output', type=Path, default=root / 'results/part_c')
    parser.add_argument('--beta-m', type=float, default=50)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.demo:
        data = json.loads((root / 'examples/prepared/demo.json').read_text())
        result = decode(data['observations'], data['arcs'])
        result['synthetic'] = True
        (args.output / 'synthetic_demo.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('Synthetic route:', result['predicted_route'])
        return
    import joblib
    from api.service import predict_trip
    from pathsnap import io
    from pathsnap.config import load_config
    model, cfg = joblib.load(args.model), load_config()
    summary = []
    for trip_id in args.trip or io.trip_ids(args.split):
        if trip_id not in io.trip_ids(args.split):
            parser.error('Unknown trip id')
        result = predict_trip(io.load_trip(args.split, trip_id), model, cfg, Config(beta_m=args.beta_m))
        result['split'] = args.split
        (args.output / f'{args.split}_{trip_id}.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        summary.append({k: result[k] for k in ['trip_id', 'split', 'metrics', 'baseline_metrics', 'timings', 'decoder_config']})
        print(trip_id, result['metrics'])
    (args.output / f'{args.split}_comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
