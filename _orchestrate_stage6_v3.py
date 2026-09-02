import os, sys, time, json, subprocess
from pathlib import Path
import pandas as pd
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.layer_probe_pipeline import load_layer_probe_config
from src.layer_probe_walk_forward import merge_walk_forward_probe_shards, validate_walk_forward_probe_outputs
config = load_layer_probe_config(ROOT/"configs"/"layer_probe_walk_forward.yaml")
t = pd.read_parquet("artifacts/probe_dataset_walk_forward_v1/probe_targets.parquet", columns=["task_id"])
tasks = sorted(t.task_id.unique().tolist())
(ROOT/"_all_tasks.json").write_text(json.dumps(tasks))
print(f"all tasks: {len(tasks)}", flush=True)
PYBIN = "/home/intern_fjq_2026/miniconda3/envs/nlp_fjq/bin/python"
env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
procs=[]; tags=[]; t0=time.time()
for layer in range(13):
    tag=str(layer); tags.append(tag)
    (ROOT/f"_lyr_{tag}.json").write_text(json.dumps([layer]))
    log=open(f"/tmp/stage6_lyr{tag}.log","w")
    p=subprocess.Popen([PYBIN,"_run_shard2.py",tag,str(ROOT/"_all_tasks.json"),str(ROOT/f"_lyr_{tag}.json")],
                       env=env,stdout=log,stderr=subprocess.STDOUT)
    procs.append(p); print(f"launched layer {layer} pid={p.pid}",flush=True)
for i,p in enumerate(procs):
    rc=p.wait(); print(f"layer {i} rc={rc} ({(time.time()-t0)/60:.1f}min)",flush=True)
    if rc: print(f"ABORT layer {i}",flush=True); sys.exit(1)
print("merging 13 layer-shards...",flush=True)
merged=merge_walk_forward_probe_shards(config,tags)
print(validate_walk_forward_probe_outputs(merged),flush=True)
print(pd.read_csv(merged/"walk_forward_selected_alphas.csv").to_string(index=False),flush=True)
print(f"[orch] total {(time.time()-t0)/60:.1f}min",flush=True)
