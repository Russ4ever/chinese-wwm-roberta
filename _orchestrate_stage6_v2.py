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
t = t[t.target_weight>0].copy(); t["feature_available_date"]=pd.to_datetime(t["feature_available_date"])
folds=[2018,2019,2020,2021,2022]
sizes={task:int(sum((tt.feature_available_date.dt.year<yr).sum() for yr in folds)) for task,tt in t.groupby("task_id")}
N_GROUPS=3
ts=sorted(sizes,key=lambda k:-sizes[k])
groups=[[] for _ in range(N_GROUPS)]; loads=[0]*N_GROUPS
for task in ts:
    i=min(range(N_GROUPS),key=lambda k:loads[k]); groups[i].append(task); loads[i]+=sizes[task]
for i,g in enumerate(groups): print(f"task-group {i}: load={loads[i]:,} n={len(g)} {g}",flush=True)
PYBIN="/home/intern_fjq_2026/miniconda3/envs/nlp_fjq/bin/python"
env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
tags=[]; procs=[]; t0=time.time()
for li in range(13):
    for gi in range(N_GROUPS):
        tag=f"g{gi}l{li}"; tags.append(tag)
        (ROOT/f"_tl_{tag}.json").write_text(json.dumps(groups[gi]))
        (ROOT/f"_tll_{tag}.json").write_text(json.dumps([li]))
        log=open(f"/tmp/stage6_{tag}.log","w")
        p=subprocess.Popen([PYBIN,"_run_shard2.py",tag,str(ROOT/f"_tl_{tag}.json"),str(ROOT/f"_tll_{tag}.json")],
                           env=env,stdout=log,stderr=subprocess.STDOUT)
        procs.append(p)
print(f"launched {len(procs)} workers",flush=True)
for i,p in enumerate(procs):
    rc=p.wait()
    if rc: print(f"ABORT worker {i}({tags[i]}) rc={rc} ({(time.time()-t0)/60:.1f}min)",flush=True); sys.exit(1)
print(f"all {len(procs)} workers done ({(time.time()-t0)/60:.1f}min), merging...",flush=True)
merged=merge_walk_forward_probe_shards(config,tags)
print(validate_walk_forward_probe_outputs(merged),flush=True)
print(pd.read_csv(merged/"walk_forward_selected_alphas.csv").to_string(index=False),flush=True)
print(f"[orch] total {(time.time()-t0)/60:.1f}min",flush=True)
