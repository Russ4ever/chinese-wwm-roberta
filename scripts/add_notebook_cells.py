"""向 activation_rank_pipeline.ipynb 添加 Cell 9 (Path C) 和 Cell 10 (Path A)。"""
import json
from pathlib import Path

NB_PATH = Path("/home/intern_fjq_2026/Projects/chinese-wwm-roberta/notebooks/activation_rank_pipeline.ipynb")

CELL_9_SOURCE = r'''# 9. 谱衰减与有效维度对比 — 复现论文 Figure 3a
# 从已有 rank_metrics + subspaces 提取指标: attention_output vs MLP vs residual
import numpy as np
import matplotlib.pyplot as plt

try:
    completed_dir
except NameError:
    completed_dir = RUN_DIR

subspaces = np.load(completed_dir / 'analysis' / 'subspaces.npz', allow_pickle=False)
rank_metrics_full = pd.read_parquet(completed_dir / 'analysis' / 'rank_metrics.parquet')
rm = rank_metrics_full[rank_metrics_full['stream'] == 'token_natural_filtered'].copy()
rm_layers = rm[rm['layer'] > 0]

summary = rm_layers.groupby('kind').agg(
    effective_rank=('effective_rank', 'mean'),
    normalized_erank=('normalized_effective_rank', 'mean'),
    k90=('k90_variance', 'mean'),
    k95=('k95_variance', 'mean'),
    k99=('k99_variance', 'mean'),
    dirs_above_1pct=('directions_above_1pct_max', 'mean'),
    dimension=('dimension', 'first'),
).round(1)
display(summary)

layer = 6
sites = {
    'Attention Output': f'token_natural_filtered__attention_output_{layer:02d}',
    'MLP Output': f'token_natural_filtered__mlp_output_{layer:02d}',
    'Residual Stream': f'token_natural_filtered__residual_{layer:02d}',
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 左: 奇异值谱 (Figure 3a)
ax = axes[0]
for label, prefix in sites.items():
    eigvals = subspaces[f'{prefix}__eigenvalues']
    sv = np.sqrt(np.maximum(eigvals, 0))
    rel_sv = sv / sv.max()
    ax.semilogy(range(len(rel_sv)), rel_sv, label=label, linewidth=2)
ax.set_xlabel('Singular Value Rank Index')
ax.set_ylabel('Relative Singular Value')
ax.set_title(f'Singular Value Spectra (Layer {layer})')
ax.legend()
ax.set_ylim(1e-5, 2)

# 中: 各层有效秩
ax = axes[1]
colors = {'attention_output': 'tab:blue', 'mlp_output': 'tab:orange', 'residual': 'tab:green'}
for kind in ['attention_output', 'mlp_output', 'residual']:
    sub_df = rm_layers[rm_layers['kind'] == kind].sort_values('layer')
    ax.plot(sub_df['layer'], sub_df['normalized_effective_rank'], 'o-',
            label=kind.replace('_', ' ').title(), color=colors[kind], linewidth=2)
ax.set_xlabel('Layer')
ax.set_ylabel('Fraction of Effective Rank')
ax.set_title('Effective Rank by Activation Type')
ax.legend()
ax.set_ylim(0, 1)

# 右: 累积方差恢复
ax = axes[2]
for label, prefix in sites.items():
    eigvals = np.maximum(subspaces[f'{prefix}__eigenvalues'], 0)
    cum = np.cumsum(eigvals) / eigvals.sum()
    ax.plot(range(1, len(cum) + 1), cum, label=label, linewidth=2)
ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='99% variance')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Variance Captured')
ax.set_title(f'Variance Recovery (Layer {layer})')
ax.legend()
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 768)

plt.tight_layout()
plt.show()

print(f'\n关键数值 (Layer {layer}):')
for label, prefix in sites.items():
    eigvals = np.maximum(subspaces[f'{prefix}__eigenvalues'], 0)
    sv = np.sqrt(eigvals)
    n_above_1pct = int((sv > 0.01 * sv.max()).sum())
    cum = np.cumsum(eigvals) / eigvals.sum()
    n_99 = int(np.searchsorted(cum, 0.99) + 1)
    dim = len(eigvals)
    print(f'  {label:20s}: dirs>1%={n_above_1pct}/{dim} ({100*n_above_1pct/dim:.1f}%), '
          f'99% var in {n_99}/{dim} ({100*n_99/dim:.1f}%)')'''

CELL_10_SOURCE = r'''# 10. 激活消融与损失恢复 — 复现论文 Figure 3b
# 零消融 → SVD 分量逐步恢复 → 下游 KL 恢复曲线
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast
from src.models.modeling import build_candidate
from src.layer_probe_representations import freeze_and_validate_inference_model

device = 'cuda:0'
compute_dtype = torch.float16

# 加载模型
candidate = build_candidate(
    base_model_dir=str(ROOT / config['model']['base_model_dir']),
    checkpoint_path=str(ROOT / config['model']['checkpoint']),
    device=device, dtype=compute_dtype,
)
freeze_and_validate_inference_model(candidate, expected_hidden_layers=13)
candidate.eval()
bert = candidate.bert

tokenizer = BertTokenizerFast.from_pretrained(
    str(ROOT / config['model']['base_model_dir']), local_files_only=True)

# 加载 500 条文本
sample = pd.read_parquet(completed_dir / 'sample' / 'sample_manifest.parquet')
reports_path = Path(str(config['text']['path']))
if not reports_path.is_absolute():
    reports_path = ROOT / reports_path
id_col = config['text'].get('id_column', 'report_id')
text_col = config['text'].get('text_column', 'text')
reports = pd.read_parquet(reports_path, columns=[id_col, text_col]).rename(
    columns={id_col: 'report_id', text_col: 'text'})
merged = sample.merge(reports[['report_id', 'text']], on='report_id', how='left')
texts = merged['text'].dropna().tolist()[:500]

encoded = tokenizer(texts, padding=True, truncation=True, max_length=512,
                     return_tensors='pt', return_token_type_ids=True)
encoded = {k: v.to(device) for k, v in encoded.items()}

# 基线前向
with torch.inference_mode():
    out = candidate(**encoded)
    baseline_logits = out.logits.float().cpu()
baseline_probs = F.softmax(baseline_logits, dim=-1)

# hook 工厂
def make_zero_hook():
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        result = torch.zeros_like(t)
        return (result,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else result
    return hook

def make_projection_hook(eigvecs, mean, n, device):
    V_n = torch.from_numpy(eigvecs[:, :n]).to(device).float()
    mean_t = torch.from_numpy(mean).to(device).float()
    P = V_n @ V_n.T
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        result = ((t.float() - mean_t) @ P + mean_t).to(t.dtype)
        return (result,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else result
    return hook

# 消融实验
sub = subspaces
experiment_sites = {'Attention Output': 'attention_output_06', 'Residual Stream': 'residual_06'}
n_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640, 768]
all_results = []

for label, site in experiment_sites.items():
    prefix = f'token_natural_filtered__{site}'
    eigvecs = sub[f'{prefix}__eigenvectors']
    mean = sub[f'{prefix}__mean']
    site_type = site.rsplit('_', 1)[0]
    layer_num = int(site.rsplit('_', 1)[1])
    layer_idx = layer_num - 1
    if site_type == 'attention_output':
        target = bert.encoder.layer[layer_idx].attention.output.dense
    elif site_type == 'mlp_output':
        target = bert.encoder.layer[layer_idx].output.dense
    elif site_type == 'residual' and layer_num > 0:
        target = bert.encoder.layer[layer_idx]
    else:
        target = bert.embeddings

    for n in n_list:
        if n == 0:
            h = make_zero_hook()
        elif n >= 768:
            h = None
        else:
            h = make_projection_hook(eigvecs, mean, n, device)
        handle = target.register_forward_hook(h) if h is not None else None
        with torch.inference_mode():
            out = candidate(**encoded)
            mod_logits = out.logits.float().cpu()
        if handle is not None:
            handle.remove()
        mod_probs = F.softmax(mod_logits, dim=-1)
        kl = F.kl_div(torch.log(mod_probs + 1e-8), baseline_probs, reduction='sum').item() / len(texts)
        all_results.append({'activation_type': label, 'n_components': n, 'kl_divergence': round(kl, 6)})

recovery = pd.DataFrame(all_results)
recovery['fraction_recovered'] = recovery.apply(
    lambda r: round(1 - r['kl_divergence'] / max(
        recovery[(recovery['activation_type'] == r['activation_type']) & (recovery['n_components'] == 0)]['kl_divergence'].values[0], 1e-8), 6), axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
for label in experiment_sites:
    sub_r = recovery[recovery['activation_type'] == label].sort_values('n_components')
    ax.plot(sub_r['n_components'], sub_r['fraction_recovered'], 'o-', label=label, linewidth=2, markersize=5)
ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='99% recovered')
ax.set_xlabel('Number of Components Retained', fontsize=12)
ax.set_ylabel('Fraction of Loss Recovered', fontsize=12)
ax.set_title(f'Downstream Loss Recovery (Layer 6)', fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-10, 780)
plt.tight_layout()
plt.show()

display(recovery[recovery['n_components'].isin([0, 64, 128, 256, 384, 768])].pivot(
    index='n_components', columns='activation_type', values='fraction_recovered').round(3))

print('\n99% 恢复所需维度:')
for label in experiment_sites:
    sub_r = recovery[recovery['activation_type'] == label]
    n99 = sub_r[sub_r['fraction_recovered'] >= 0.99]['n_components']
    n99 = int(n99.min()) if len(n99) > 0 else 768
    print(f'  {label}: {n99}/768 ({100*n99/768:.1f}%)')'''


def main():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    def make_cell(source, cell_id):
        return {
            "cell_type": "code",
            "id": cell_id,
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source.split("\n")]
                       if not source.endswith("\n")
                       else source.split("\n"),
        }

    # Normalize source: split into lines, add \n to each except last
    def make_cell_proper(source, cell_id):
        lines = source.split("\n")
        src_list = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        return {
            "cell_type": "code",
            "id": cell_id,
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src_list,
        }

    cell9 = make_cell_proper(CELL_9_SOURCE, "spectral-analysis")
    cell10 = make_cell_proper(CELL_10_SOURCE, "loss-recovery")

    nb["cells"].append(cell9)
    nb["cells"].append(cell10)

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Notebook 已更新: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
