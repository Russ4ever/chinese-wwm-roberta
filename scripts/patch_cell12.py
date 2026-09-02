"""Patch cell 12 of activation_rank_pipeline.ipynb with subspace coverage display."""
import json
from pathlib import Path

NB = Path("notebooks/activation_rank_pipeline.ipynb")
CELL_SOURCE = (
    "# 12. Attention output 写入子空间覆盖分析\n"
    "# 纯线性代数: 各层 top-K 子空间的重合度、累计覆盖谱、低覆盖方向\n"
    "# 不加载模型, 不读取标签, 仅复用上游 SVD 结果\n"
    "from IPython.display import Image\n"
    "\n"
    "from src.activation_subspace_coverage import run_coverage_stage\n"
    "\n"
    "COVERAGE_POLICY = ROOT / 'configs' / 'activation_subspace_coverage.yaml'\n"
    "coverage_dir = run_coverage_stage(config, COVERAGE_POLICY)\n"
    "\n"
    "coverage_manifest = json.loads(\n"
    "    (coverage_dir / 'manifest.json').read_text(encoding='utf-8')\n"
    ")\n"
    "pairwise_df = pd.read_parquet(coverage_dir / 'pairwise_overlap.parquet')\n"
    "cumulative_df = pd.read_parquet(coverage_dir / 'cumulative_coverage.parquet')\n"
    "eigvals_df = pd.read_parquet(coverage_dir / 'coverage_eigenvalues.parquet')\n"
    "summary_df = pd.read_parquet(coverage_dir / 'coverage_summary.parquet')\n"
    "invariants_df = pd.read_parquet(coverage_dir / 'numerical_invariants.parquet')\n"
    "\n"
    "display({\n"
    "    'output': str(coverage_dir),\n"
    "    'scope': coverage_manifest['scope'],\n"
    "    'k_grid': coverage_manifest['k_grid'],\n"
    "    'primary_k': coverage_manifest['primary_k'],\n"
    "    'layers': coverage_manifest['layers'],\n"
    "    'random_overlap_formula': coverage_manifest['random_overlap_formula'],\n"
    "    'labels_or_returns_loaded': coverage_manifest['labels_or_returns_loaded'],\n"
    "})\n"
    "\n"
    "# 数值不变量\n"
    "display(invariants_df)\n"
    "\n"
    "# 各 K 的 coverage summary\n"
    "display(summary_df)\n"
    "\n"
    "# 关键: K=256 (primary) 的 pairwise excess overlap\n"
    "pk = summary_df['k'].min()  # primary_k from config\n"
    "primary_k = coverage_manifest['primary_k']\n"
    "excess_at_pk = pairwise_df[\n"
    "    pairwise_df['k'] == primary_k\n"
    "][['layer_i', 'layer_j', 'layer_distance', 'overlap',\n"
    "    'random_overlap_baseline', 'excess_overlap']].sort_values(\n"
    "    'excess_overlap', ascending=False)\n"
    "display(excess_at_pk.head(20))\n"
    "\n"
    "# 累计覆盖增长\n"
    "display(cumulative_df[['k', 'prefix_layers', 'effective_coverage_rank',\n"
    "    'directions_coverage_below_005', 'directions_coverage_above_050']])\n"
    "\n"
    "# 三张图\n"
    "display(Image(filename=str(coverage_dir / 'pairwise_overlap_heatmap.png')))\n"
    "display(Image(filename=str(coverage_dir / 'cumulative_coverage.png')))\n"
    "display(Image(filename=str(coverage_dir / 'coverage_spectrum.png')))\n"
)

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Ensure cell 12 exists
while len(nb["cells"]) < 12:
    nb["cells"].append({
        "cell_type": "code", "id": f"cell-{len(nb['cells'])}",
        "metadata": {}, "execution_count": None, "outputs": [], "source": [],
    })

lines = CELL_SOURCE.strip().split("\n")
nb["cells"][11]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
nb["cells"][11]["outputs"] = []
nb["cells"][11]["execution_count"] = None
nb["cells"][11]["id"] = "subspace-coverage"

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Cell 12 patched with subspace coverage display")
