"""Patch train_subspace_fc_heads: use sum_cls directly, no division by n_posts."""
import json
from pathlib import Path

NB = Path("notebooks/train_subspace_fc_heads.ipynb")
with open(NB.open("r", encoding="utf-8")) as f:
    nb = json.load(f)

# Cell 1 (index 1): replace mean_cls with sum_cls_feat
src1 = "".join(nb["cells"][1]["source"])

src1 = src1.replace(
    "# 1. 加载 sum_cls + 投影到 H/L + 收益标签 + 滚动切分",
    "# 1. 加载 sum_cls (不除n_posts) + 投影到 H/L + 收益标签 + 滚动切分",
)

OLD_MEAN = "mean_cls = (np.stack(pf[\"sum_cls\"].values).astype(np.float64) / nposts_safe[:, None]).astype(np.float32)"
NEW_SUM = "sum_cls_feat = np.stack(pf[\"sum_cls\"].values).astype(np.float32)  # sum, 不除以 n_posts"
assert OLD_MEAN in src1, f"mean_cls line not found"
src1 = src1.replace(OLD_MEAN, NEW_SUM)

src1 = src1.replace('print(f"mean_cls: {mean_cls.shape}', 'print(f"sum_cls: {sum_cls_feat.shape}')
src1 = src1.replace("feat_H = mean_cls @ Q_H", "feat_H = sum_cls_feat @ Q_H")
src1 = src1.replace("feat_L = mean_cls @ Q_L", "feat_L = sum_cls_feat @ Q_L")

lines1 = src1.strip().split("\n")
nb["cells"][1]["source"] = [l + "\n" for l in lines1[:-1]] + [lines1[-1]]
nb["cells"][1]["outputs"] = []
nb["cells"][1]["execution_count"] = None

# markdown title
nb["cells"][0]["source"][0] = "# 训练子空间 fc head (H / L) — sum_cls 直接训练 + OOS\n"

# Clear all outputs and execution counts
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(NB.open("w", encoding="utf-8")) as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Verify
src = "".join(nb["cells"][1]["source"])
assert "mean_cls" not in src, "mean_cls still present!"
assert "sum_cls_feat" in src, "sum_cls_feat not found!"
assert "sum_cls_feat @ Q_H" in src, "feat_H not using sum_cls_feat!"
assert "sum_cls_feat @ Q_L" in src, "feat_L not using sum_cls_feat!"
print("Patched OK:")
print("  - mean_cls removed")
print("  - sum_cls_feat used directly for feat_H and feat_L")
print("  - no division by n_posts")
