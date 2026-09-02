"""Patch cell 10 of activation_rank_pipeline.ipynb with CLS representation recovery source."""
import json
from pathlib import Path

NB = Path("notebooks/activation_rank_pipeline.ipynb")
CELL_SOURCE = (
    "# 10. 逐层 attention output CLS 表征恢复\n"
    "# 对每一层的 attention output 做 SVD 投影消融, 测量 CLS pooled representation\n"
    "# 与基线的 L2 距离变化; PCA 样本与评估研报按 report_id/text hash 隔离。\n"
    "from IPython.display import Image\n"
    "\n"
    "from src.activation_rank_loss_recovery import run_loss_recovery_stage\n"
    "\n"
    "LOSS_RECOVERY_POLICY = ROOT / 'configs' / 'activation_rank_loss_recovery.yaml'\n"
    "loss_recovery_dir = run_loss_recovery_stage(config, LOSS_RECOVERY_POLICY)\n"
    "\n"
    "loss_recovery_manifest = json.loads(\n"
    "    (loss_recovery_dir / 'manifest.json').read_text(encoding='utf-8')\n"
    ")\n"
    "loss_recovery_summary = pd.read_parquet(loss_recovery_dir / 'site_summary.parquet')\n"
    "loss_recovery_metrics = pd.read_parquet(\n"
    "    loss_recovery_dir / 'loss_recovery_metrics.parquet'\n"
    ")\n"
    "\n"
    "display({\n"
    "    'output': str(loss_recovery_dir),\n"
    "    'loss_semantics': loss_recovery_manifest['loss_semantics'],\n"
    "    'intervention_scope': loss_recovery_manifest['intervention_scope'],\n"
    "    'layers': loss_recovery_manifest['layers'],\n"
    "    'kinds': loss_recovery_manifest['kinds'],\n"
    "    'evaluation_reports': loss_recovery_manifest['evaluation_reports'],\n"
    "    'batch_size': loss_recovery_manifest['batch_size'],\n"
    "    'pca_sample_report_overlap': loss_recovery_manifest['pca_sample_report_overlap'],\n"
    "    'pca_sample_text_hash_overlap': loss_recovery_manifest['pca_sample_text_hash_overlap'],\n"
    "})\n"
    "display(loss_recovery_summary)\n"
    "display(\n"
    "    loss_recovery_metrics.loc[\n"
    "        loss_recovery_metrics['n_components'].isin([0, 64, 128, 256, 384, 512, 768]),\n"
    "        [\n"
    "            'site', 'layer', 'n_components', 'distance_zero',\n"
    "            'distance_projected', 'status', 'recovery_fraction',\n"
    "            'recovery_ci_lower', 'recovery_ci_upper',\n"
    "        ],\n"
    "    ]\n"
    ")\n"
    "display(Image(filename=str(loss_recovery_dir / 'loss_recovery.png')))\n"
)

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

lines = CELL_SOURCE.strip().split("\n")
nb["cells"][9]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
nb["cells"][9]["outputs"] = []
nb["cells"][9]["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Cell 10 patched with proper quoting")
