"""Idempotently install the audited Cell 10 orchestration code."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "activation_rank_pipeline.ipynb"


CELL_10_SOURCE = r"""# 10. 研报域单位置激活干预与 MLM-proxy 损失恢复
# 三类 activation 均只干预 Layer 6；PCA 样本与评估研报按 report_id/text hash 隔离。
# 该损失使用“微调 backbone + base-model MLM head”，不是 checkpoint 原训练目标。
from IPython.display import Image

from src.activation_rank_loss_recovery import run_loss_recovery_stage

LOSS_RECOVERY_POLICY = ROOT / 'configs' / 'activation_rank_loss_recovery.yaml'
loss_recovery_dir = run_loss_recovery_stage(config, LOSS_RECOVERY_POLICY)

loss_recovery_manifest = json.loads(
    (loss_recovery_dir / 'manifest.json').read_text(encoding='utf-8')
)
loss_recovery_summary = pd.read_parquet(loss_recovery_dir / 'site_summary.parquet')
loss_recovery_metrics = pd.read_parquet(
    loss_recovery_dir / 'loss_recovery_metrics.parquet'
)

display({
    'output': str(loss_recovery_dir),
    'loss_semantics': loss_recovery_manifest['loss_semantics'],
    'intervention_scope': loss_recovery_manifest['intervention_scope'],
    'evaluation_reports': loss_recovery_manifest['evaluation_reports'],
    'mask_seeds': loss_recovery_manifest['mask_seeds'],
    'pca_sample_report_overlap': loss_recovery_manifest['pca_sample_report_overlap'],
    'pca_sample_text_hash_overlap': loss_recovery_manifest['pca_sample_text_hash_overlap'],
})
display(loss_recovery_summary)
display(
    loss_recovery_metrics.loc[
        loss_recovery_metrics['n_components'].isin([0, 64, 128, 256, 384, 512, 768]),
        [
            'site', 'n_components', 'loss_original', 'loss_zero',
            'loss_projected', 'status', 'recovery_fraction',
            'recovery_ci_lower', 'recovery_ci_upper',
        ],
    ]
)
display(Image(filename=str(loss_recovery_dir / 'loss_recovery.png')))
"""


def main() -> None:
    notebook = nbformat.read(NB_PATH, as_version=4)
    replacement = nbformat.v4.new_code_cell(
        source=CELL_10_SOURCE,
        metadata={},
        execution_count=None,
        outputs=[],
    )
    replacement["id"] = "loss-recovery"
    matches = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.get("id") == "loss-recovery"
    ]
    if len(matches) > 1:
        raise RuntimeError("notebook中存在多个loss-recovery cell")
    if matches:
        notebook.cells[matches[0]] = replacement
    else:
        notebook.cells.append(replacement)
    nbformat.write(notebook, NB_PATH)
    print(f"Notebook Cell 10已更新: {NB_PATH}")


if __name__ == "__main__":
    main()
