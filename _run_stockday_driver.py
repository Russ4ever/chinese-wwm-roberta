import os, sys, time
from pathlib import Path
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.layer_probe_pipeline import load_layer_probe_config
from src.layer_probe_panel import run_stock_day_panel_stage, validate_stock_day_artifacts
config = load_layer_probe_config(ROOT/"configs"/"layer_probe_walk_forward.yaml")
print("[stock_day] start", flush=True)
t0=time.time()
d = run_stock_day_panel_stage(config, "validation")
print(f"[stock_day] DONE in {(time.time()-t0)/60:.1f} min -> {d}", flush=True)
print(validate_stock_day_artifacts(d, evaluation_split="validation"), flush=True)
