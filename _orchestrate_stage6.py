import os, sys, time, json, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.layer_probe_pipeline import load_layer_probe_config
from src.layer_probe_walk_forward import merge_walk_forward_probe_shards, validate_walk_forward_probe_outputs
config = load_layer_probe_config(ROOT/"configs"/"layer_probe_walk_forward.yaml")
t = pd.read_parquet("artifacts/probe_dataset_walk_forward_v1/probe_targets.parquet",
                    columns=["task_id","feature_available_date","target_weight"])
t = t[t.target_weight>0].copy()
t["feature_available_date"] = pd.to_datetime(t["feature_available_date"])
folds=[2018,2019,2020,2021,2022]
sizes = {task: int(sum((tt.feature_available_date.dt.year<yr).sum() for yr in folds))
         for task, tt in t.groupby("task_id")}
N_SHARDS = 8
tasks_sorted = sorted(sizes, key=lambda k: -sizes[k])
bins = [[] for _ in range(N_SHARDS)]; loads=[0]*N_SHARDS
for task in tasks_sorted:
    i = min(range(N_SHARDS), key=lambda k: loads[k])
    bins[i].append(task); loads[i] += sizes[task]
for i,b in enumerate(bins):
    (ROOT/f"_shard{i}_tasks.json").write_text(json.dumps(b))
    print(f"bin {i}: load={loads[i]:,} ntasks={len(b)} tasks={b}", flush=True)
PYBIN = "/home/intern_fjq_2026/miniconda3/envs/nlp_fjq/bin/python"
env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
procs=[]
t0=time.time()
for i in range(N_SHARDS):
    log = open(f"/tmp/stage6_shard{i}.log","w")
    p = subprocess.Popen([PYBIN, str(ROOT/"_run_shard.py"), str(i), str(ROOT/f"_shard{i}_tasks.json")],
                         env=env, stdout=log, stderr=subprocess.STDOUT)
    procs.append(p); print(f"launched shard {i} pid={p.pid}", flush=True)
for i,p in enumerate(procs):
    rc = p.wait()
    print(f"shard {i} pid={p.pid} rc={rc} ({(time.time()-t0)/60:.1f}min)", flush=True)
    if rc != 0: print(f"ABORT: shard {i} FAILED rc={rc}", flush=True); sys.exit(1)
print("all shards done, merging...", flush=True)
merged = merge_walk_forward_probe_shards(config, [str(i) for i in range(N_SHARDS)])
print(validate_walk_forward_probe_outputs(merged), flush=True)
print(pd.read_csv(merged/"walk_forward_selected_alphas.csv").to_string(index=False), flush=True)
print(f"[orchestrator] total {(time.time()-t0)/60:.1f} min", flush=True)
