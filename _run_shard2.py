import os, sys, time, json
from pathlib import Path
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.layer_probe_pipeline import load_layer_probe_config
from src.layer_probe_walk_forward import run_walk_forward_probe_stage
shard_tag = sys.argv[1]; tasks_json = sys.argv[2]; layers_json = sys.argv[3]
task_ids = json.loads(Path(tasks_json).read_text())
layers = json.loads(Path(layers_json).read_text())
config = load_yaml = load_layer_probe_config(ROOT/"configs"/"layer_probe_walk_forward.yaml")
t0 = time.time()
print(f"[{shard_tag}] start tasks={len(task_ids)} layers={layers}", flush=True)
d = run_walk_forward_probe_stage(config, task_ids=task_ids, layers=layers, shard_tag=shard_tag)
print(f"[{shard_tag}] DONE {(time.time()-t0)/60:.1f}min -> {d}", flush=True)
