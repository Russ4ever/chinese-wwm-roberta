import os, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta")
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from src.layer_probe_pipeline import load_layer_probe_config
from src.layer_probe_walk_forward import (
    parse_walk_forward_protocol, run_walk_forward_probe_stage,
    validate_walk_forward_probe_outputs,
)
config = load_layer_probe_config(ROOT / "configs" / "layer_probe_walk_forward.yaml")
protocol = parse_walk_forward_protocol(config)
print("[stage6] config loaded; run_dir=", config["output"]["run_directory"], flush=True)
t0 = time.time()
d = run_walk_forward_probe_stage(config)
print(f"[stage6] DONE in {(time.time()-t0)/60:.1f} min -> {d}", flush=True)
print(validate_walk_forward_probe_outputs(d), flush=True)
print(pd.read_csv(d / "walk_forward_selected_alphas.csv").to_string(index=False), flush=True)
