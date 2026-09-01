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

CELL_10_SOURCE = r'''# 10. 激活消融与损失恢复 — 复现论文 Figure 3b (MLM CE, 论文公式)
# 与论文严格一致: 下游损失 = MLM 交叉熵; 投影作用于原始激活 (N=0 == 零消融)
# fraction = (loss_zero - loss_N) / (loss_zero - loss_original)
# BERT 残差连接冗余度高, 单层 attention 消融对 MLM 损失影响≈0, 故 attention/MLP
# 采用 12 层联合消融; residual stream 保持 layer 6 单层消融 (分母天然健康)。
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import BertTokenizerFast, BertForMaskedLM
from src.models.checkpoint import load_state_dict_safe, strip_prefix

device = 'cuda:0'

# 加载模型: 微调 backbone + base MLM head
base_dir = str(ROOT / config['model']['base_model_dir'])
mlm = BertForMaskedLM.from_pretrained(base_dir)
state = strip_prefix(load_state_dict_safe(str(ROOT / config['model']['checkpoint']), map_location='cpu'))
bert_state = {k[len('bert.'):]: v for k, v in state.items()
              if k.startswith('bert.') and not k.startswith('bert.pooler.')}
mlm.bert.load_state_dict(bert_state, strict=True)
mlm = mlm.to(device=device, dtype=torch.float16).eval()
bert = mlm.bert

tokenizer = BertTokenizerFast.from_pretrained(base_dir, local_files_only=True)

# 加载 500 条文本 (用 RUN_DIR, 不依赖 cell 8 的 completed_dir)
sample = pd.read_parquet(RUN_DIR / 'sample' / 'sample_manifest.parquet')
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

# 固定 mask (所有条件共用, 保证可比)
generator = torch.Generator(device=device).manual_seed(20260901)
special = ((encoded['input_ids'] == tokenizer.cls_token_id)
           | (encoded['input_ids'] == tokenizer.sep_token_id)
           | (encoded['input_ids'] == tokenizer.pad_token_id))
prob = torch.rand(encoded['input_ids'].shape, device=device, generator=generator)
mask_positions = (prob < 0.15) & ~special
masked_ids = encoded['input_ids'].clone()
masked_ids[mask_positions] = tokenizer.mask_token_id
targets = encoded['input_ids'][mask_positions]

def masked_lm_loss():
    out = mlm(input_ids=masked_ids, attention_mask=encoded['attention_mask'],
              token_type_ids=encoded['token_type_ids'])
    logits = out.logits[mask_positions].float()
    return F.cross_entropy(logits, targets).item()

with torch.inference_mode():
    loss_original = masked_lm_loss()

# hook: 非中心化投影, n=0 时 P=0 -> 全零 (零消融)
def make_projection_hook(eigvecs, n):
    V_n = torch.from_numpy(eigvecs[:, :n]).to(device).float()
    P = V_n @ V_n.T
    def hook(_m, _i, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        result = (t.float() @ P).to(t.dtype)
        return (result,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else result
    return hook

def resolve_target(site):
    site_type = site.rsplit('_', 1)[0]
    layer_num = int(site.rsplit('_', 1)[1])
    layer_idx = layer_num - 1
    if site_type == 'attention_output':
        return bert.encoder.layer[layer_idx].attention.output.dense
    if site_type == 'mlp_output':
        return bert.encoder.layer[layer_idx].output.dense
    if site_type == 'residual' and layer_num > 0:
        return bert.encoder.layer[layer_idx]
    if site_type == 'residual' and layer_num == 0:
        return bert.embeddings
    if site_type == 'z':
        return bert.encoder.layer[layer_idx].attention.self
    raise ValueError(site)

sub = subspaces
experiment_sites = {
    'Attention Output': [f'attention_output_{i:02d}' for i in range(1, 13)],
    'MLP Output': [f'mlp_output_{i:02d}' for i in range(1, 13)],
    'Residual Stream': ['residual_06'],
}
n_list = [0, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 736, 768]
all_results = []

for label, site_list in experiment_sites.items():
    module_eigvecs = []
    for site in site_list:
        prefix = f'token_natural_filtered__{site}'
        module_eigvecs.append((resolve_target(site), sub[f'{prefix}__eigenvectors']))

    for n in n_list:
        handles = []
        if n < 768:
            for target, eigvecs in module_eigvecs:
                handles.append(target.register_forward_hook(make_projection_hook(eigvecs, n)))
        with torch.inference_mode():
            loss_n = masked_lm_loss()
        for h in handles:
            h.remove()
        all_results.append({'activation_type': label, 'n_components': n, 'ce_loss': round(loss_n, 6)})

results_df = pd.DataFrame(all_results)
recovery = []
for label in experiment_sites:
    sub_df = results_df[results_df['activation_type'] == label].set_index('n_components')
    loss_zero = sub_df.loc[0, 'ce_loss']
    denom = max(loss_zero - loss_original, 1e-8)
    for n in n_list:
        loss_n = sub_df.loc[n, 'ce_loss']
        recovery.append({
            'activation_type': label, 'n_components': n, 'ce_loss': loss_n,
            'fraction_recovered': round((loss_zero - loss_n) / denom, 6),
        })
recovery_df = pd.DataFrame(recovery)

fig, ax = plt.subplots(figsize=(9, 5))
for label in experiment_sites:
    sub_r = recovery_df[recovery_df['activation_type'] == label].sort_values('n_components')
    ax.plot(sub_r['n_components'], sub_r['fraction_recovered'], 'o-',
            label=label, linewidth=2, markersize=5)
ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='99% recovered')
ax.set_xlabel('Number of Components Retained (per site)', fontsize=12)
ax.set_ylabel('Fraction of Loss Recovered', fontsize=12)
ax.set_title('Downstream MLM Loss Recovery (Attention/MLP: L1-12 joint; Residual: L6)', fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-10, 780)
plt.tight_layout()
plt.show()

display(recovery_df[recovery_df['n_components'].isin([0, 64, 128, 256, 384, 512, 768])].pivot(
    index='n_components', columns='activation_type', values='fraction_recovered').round(3))

print(f'\nloss_original={loss_original:.4f}  99% 恢复所需维度:')
for label in experiment_sites:
    sub_r = recovery_df[recovery_df['activation_type'] == label]
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

    # 幂等: 已存在同 id 的 cell 则原位替换, 否则追加
    existing = {c.get("id"): idx for idx, c in enumerate(nb["cells"])}
    for cell in (cell9, cell10):
        if cell["id"] in existing:
            nb["cells"][existing[cell["id"]]] = cell
        else:
            nb["cells"].append(cell)

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Notebook 已更新: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
