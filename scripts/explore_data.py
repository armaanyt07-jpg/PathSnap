#!/usr/bin/env python3
"""Reproducible Part A EDA. Writes trip-level CSV and split summary JSON."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap import io
from pathsnap.config import load_config, results_dir

def main():
    cfg = load_config(); rows=[]; split_summary={}
    for split in cfg["splits"].values():
        for tid in io.trip_ids(split):
            base=io.trip_base(split,tid); trip=io.load_trip(split,tid)
            tr=trip["track"]; nodes=trip["nodes"]; arcs=trip["arcs"]; route=trip["route"]
            ts=tr.t.to_numpy(float); lon=tr.lon.to_numpy(float); lat=tr.lat.to_numpy(float)
            dt=np.diff(ts); valid=np.isfinite(lon)&np.isfinite(lat)&np.isfinite(ts)
            endpoints=set(arcs.reshape(-1).tolist()) if arcs.size else set()
            degrees=np.bincount(arcs.reshape(-1), minlength=len(nodes)) if arcs.size else np.zeros(len(nodes),int)
            # Weakly connected components via union-find over directed arcs.
            parent=np.arange(len(nodes))
            def find(x):
                while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
                return x
            for a,b in arcs:
                ra,rb=find(int(a)),find(int(b))
                if ra!=rb: parent[ra]=rb
            comps=len({find(i) for i in range(len(nodes))}) if len(nodes) else 0
            files={e:base.with_suffix(e) for e in ('.track','.nodes','.arcs','.route')}
            rows.append(dict(split=split,trip_id=tid,n_obs=len(tr),n_nodes=len(nodes),n_arcs=len(arcs),n_route=0 if route is None else len(route),
                missing_files=';'.join(e for e,p in files.items() if not p.exists()),size_mb=round(sum(p.stat().st_size for p in files.values() if p.exists())/1e6,3),
                invalid_gps=int((~valid).sum()),timestamp_nonincreasing=int((dt<=0).sum()),gap_gt_1s=int((dt>1).sum()),gap_gt_5s=int((dt>5).sum()),
                dt_median=float(np.median(dt)) if len(dt) else None,dt_min=float(np.min(dt)) if len(dt) else None,dt_max=float(np.max(dt)) if len(dt) else None,
                lon_min=float(np.nanmin(lon)) if len(lon) else None,lon_max=float(np.nanmax(lon)) if len(lon) else None,lat_min=float(np.nanmin(lat)) if len(lat) else None,lat_max=float(np.nanmax(lat)) if len(lat) else None,
                isolated_nodes=int((degrees==0).sum()),weak_components=comps,route_unique_arcs=0 if route is None else len(np.unique(route))))
        g=pd.DataFrame([r for r in rows if r['split']==split])
        split_summary[split]={"trips":len(g),"observations":int(g.n_obs.sum()),"median_observations":float(g.n_obs.median()),"trips_missing_files":int((g.missing_files!='').sum()),"invalid_gps_rows":int(g.invalid_gps.sum()),"nonincreasing_timestamp_steps":int(g.timestamp_nonincreasing.sum()),"gaps_over_1s":int(g.gap_gt_1s.sum()),"gaps_over_5s":int(g.gap_gt_5s.sum()),"median_network_components":float(g.weak_components.median()),"median_isolated_nodes":float(g.isolated_nodes.median()),"trips_without_route":int((g.n_route==0).sum()),"columns":list(g.columns)}
    out=results_dir(); pd.DataFrame(rows).to_csv(out/'eda_trips.csv',index=False)
    (out/'eda_summary.json').write_text(json.dumps(split_summary,indent=2),encoding='utf-8')
    print(f"Wrote {out/'eda_trips.csv'} and {out/'eda_summary.json'}")
if __name__=='__main__': main()
