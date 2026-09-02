import os, sys, time
from pathlib import Path
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta"); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.layer_probe_pipeline import load_layer_probe_config, validate_factor_outputs
from src.layer_probe_label_factors import (run_layer_factor_correlation_stage, validate_layer_factor_correlation_outputs)
from src.layer_probe_factors import run_factor_validation_stage
config = load_layer_probe_config(ROOT/"configs"/"layer_probe_walk_forward.yaml")
def step(name, fn):
    t0=time.time(); print(f"\n=== {name} ===", flush=True); d=fn()
    print(f"{name} DONE {(time.time()-t0)/60:.1f}min -> {d}", flush=True); return d
lc = step("layer_correlations(cell25)", lambda: run_layer_factor_correlation_stage(config))
print(validate_layer_factor_correlation_outputs(lc), flush=True)
fv = step("factor_validation(cell27)", lambda: run_factor_validation_stage(config,"validation"))
print(validate_factor_outputs(fv), flush=True)
print("\n[phase3b] ALL DONE", flush=True)
