import os, sys, time
from pathlib import Path
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.layer_probe_pipeline import load_layer_probe_config
from src.layer_probe_walk_forward import align_walk_forward_targets, validate_walk_forward_targets
config = load_layer_probe_config(ROOT/"configs"/"layer_probe_walk_forward.yaml")
t0 = time.time()
d = align_walk_forward_targets(config)
print(f"align DONE {(time.time()-t0)/60:.1f}min -> {d}", flush=True)
print(validate_walk_forward_targets(d), flush=True)
