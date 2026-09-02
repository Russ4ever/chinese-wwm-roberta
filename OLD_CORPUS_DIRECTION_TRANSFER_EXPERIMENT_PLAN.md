# 旧语料 Coverage Directions 向新语料迁移的消融实验方案

## 1. 研究定位

本实验是现有 activation-rank 研究的跨语料迁移扩展。它冻结旧金融研报语料中识别出的 attention output coverage directions，并将其直接应用于完全不同的新语料。

本实验不在新语料上重新拟合 PCA/SVD 或 coverage directions，回答的问题是：

> 旧金融研报语料中发现的高覆盖和低覆盖 attention 写入方向，在新语料上是否仍表现出可迁移的功能重要性？

本实验不能用于声称：

- 这些方向是新语料自身的高覆盖或低覆盖方向；
- 新语料具有与旧语料相同的 activation rank；
- 这些方向包含增量 alpha；
- 这些方向是跨层因果信息通道。

旧语料实验、新语料迁移实验和未来新语料内生方向重估必须使用不同的 run directory 和 manifest，不能覆盖或混合产物。

## 2. 固定不变的模型契约

迁移实验必须继续使用与旧语料实验完全相同的：

- checkpoint；
- tokenizer；
- pooling path；
- max length=512；
- truncation policy；
- attention output hook 位置；
- hidden dimension=768；
- eval mode；
- frozen parameters；
- inference mode。

模型默认使用 physical GPU 1。启动前必须执行共享服务器 GPU、CPU、NUMA、磁盘和进程 preflight，不得自动使用 GPU 0。

## 3. 旧语料方向的来源

旧语料 canonical 输入：

    artifacts/checkpoint_activation_rank/runs/financial_reports_v2/
    ├── analysis/
    │   ├── subspaces.npz
    │   └── manifest.json
    └── extensions/
        └── attention_subspace_coverage_v1/
            ├── coverage_eigenvalues.parquet
            └── manifest.json

当前 coverage extension 只在 primary K=256 下保存了 low/high coverage vectors，不能直接用于本方案的 K=64 和 K=640 定义。

实现必须从旧语料 subspaces.npz 重新构造 \(C_{64}\) 和 \(C_{640}\)，并通过旧 analysis/coverage manifest hash 锁定输入。

## 4. 实验 A：保留旧 K=64 的低覆盖方向

对旧语料每层 attention output 的 top-64 子空间构造：

\[
P_{l,64}=V_{l,64}V_{l,64}^{\top}.
\]

全层平均 coverage matrix：

\[
C_{64}
=
\frac1{12}\sum_{l=1}^{12}P_{l,64}.
\]

特征分解：

\[
C_{64}Q_{64}
=
Q_{64}\Gamma_{64}.
\]

定义旧语料低覆盖空间：

\[
S_A
=
\{q_r^{64}:\gamma_r^{64}<0.05\}.
\]

按照旧结果，预期：

\[
\dim(S_A)=449.
\]

实验 A 的主干预是只保留这 449 个方向：

\[
O_l^{A}
=
O_lP_{S_A}.
\]

为减少推理计算，实际实现应使用其 319 维正交补：

\[
O_l^{A}
=
O_l
-
(O_lQ_{A,\mathrm{drop}})
Q_{A,\mathrm{drop}}^{\top}.
\]

其中：

\[
\dim(Q_{A,\mathrm{drop}})=319.
\]

该干预回答：

> 如果只保留旧语料中很少进入各层 top-64 核心写入空间的方向，新语料上的最终 CLS 和冻结头输出还能保留多少？

这些 449 维不是旧语料每层最低方差的 449 个 components。它们是由全层 \(C_{64}\) 得到的低 coverage directions。

## 5. 实验 B：删除旧 K=640 的高覆盖方向

对旧语料每层 attention output 的 top-640 子空间构造：

\[
P_{l,640}=V_{l,640}V_{l,640}^{\top}.
\]

全层平均 coverage matrix：

\[
C_{640}
=
\frac1{12}\sum_{l=1}^{12}P_{l,640}.
\]

特征分解：

\[
C_{640}Q_{640}
=
Q_{640}\Gamma_{640}.
\]

定义旧语料高覆盖空间：

\[
H_B
=
\{q_r^{640}:\gamma_r^{640}\ge0.90\}.
\]

按照旧结果，预期：

\[
\dim(H_B)=320.
\]

实验 B 的主干预：

\[
O_l^{B}
=
O_l
-
(O_lQ_{B,\mathrm{high}})
Q_{B,\mathrm{high}}^{\top}.
\]

最终保留：

\[
768-320=448
\]

个方向。

该干预回答：

> 删除旧语料中几乎总能进入各层宽 top-640 写入空间的高 coverage directions，是否会对新语料最终表征造成超过随机方向的功能破坏？

## 6. 冻结旧方向 artifact

实现必须先生成独立、不可变的方向 artifact：

    old_corpus_transfer_direction_sets_v1.npz

至少保存：

    experiment_a_keep_vectors
    experiment_a_drop_vectors
    experiment_a_coverage_eigenvalues
    experiment_b_drop_vectors
    experiment_b_keep_vectors
    experiment_b_coverage_eigenvalues

同时生成方向 manifest，记录：

- 旧语料 analysis manifest SHA-256；
- 旧语料 subspaces.npz SHA-256；
- 旧语料 coverage manifest SHA-256；
- K=64 和 K=640；
- 0.05 和 0.90 阈值；
- 实际方向数量；
- 每个 projector 的 SHA-256；
- 代码和 policy SHA-256。

必须验证：

\[
Q^\top Q\approx I,
\qquad
P=P^\top,
\qquad
P^2\approx P.
\]

如果方向数量不是预期的 449 和 320，必须停止并报告 fingerprint 或协议不匹配，不能静默调整阈值。

## 7. 零推理成本的跨定义重合检查

实验 A 最终保留 449 维，实验 B 最终保留 448 维。两者可能接近同一个保留空间。

在模型推理前计算：

\[
\operatorname{overlap}(S_A,H_B^\perp)
=
\frac{
\lVert Q_A^\top Q_B\rVert_F^2
}{
\min(449,448)
}.
\]

随机基准约为：

\[
\frac{448}{768}
\approx0.5833.
\]

同时输出：

- normalized overlap；
- excess overlap；
- principal-angle cos² 的 min/mean/max 和分位数；
- 两个 drop spaces 的 overlap；
- projector Frobenius distance。

如果两个保留空间高度重合，应将实验 A 设为主实验、实验 B 设为方向定义敏感性，避免重复扩展大量随机对照。

该规则必须在查看新语料反事实结果前执行并写入 preflight。

## 8. 新语料输入契约

新语料至少需要：

- 稳定唯一的文本 ID；
- 文本列；
- 可重算的 text SHA-256；
- 可选 source/category/date 字段；
- tokenizer 后 token count。

新语料文件、文本内容和排序必须进入 run fingerprint。不能仅依赖文件路径。

旧语料方向是固定外生变换，因此描述性迁移实验可以在新语料全体分布上评估。但若后续进入 alpha 研究，必须另建严格 walk-forward 流程，不得把本实验的新语料统计量作为未来 fold 的训练外信息。

## 9. 新语料评估样本

建立两个相互独立的 deterministic evaluation manifests。

### 9.1 Pilot

- 256 份文本；
- report/text ID 唯一；
- text hash 唯一；
- 与旧 PCA sample 和旧 held-out sample 做 text-hash 重叠审计；
- 若新语料长度分布跨度大，按 token length 四分位分层；
- 若存在 source/category，保留供分组诊断。

### 9.2 Confirmation

- 1024 份文本；
- 与 Pilot 在 ID 和 text hash 上均零重叠；
- 只有 Pilot 达到预设继续条件时才运行；
- Pilot 结束后冻结所有方向、条件、随机种子、评价指标和决策规则。

如果新语料不足，应在 manifest 中记录全部可用文本和抽样原因，不能通过重复文本补足样本。

## 10. 新语料 baseline forward

第一次无干预前向同时完成：

1. 保存原始 pooled CLS；
2. 保存冻结分类头 logits；
3. 累计新语料 12 层 attention output 的均值；
4. 累计新语料 12 层 attention output 的 centered covariance；
5. 记录每层 activation norm、有效 token 数和过滤比例；
6. 为随机能量匹配提供统计量。

该阶段不在新语料上重新拟合主实验方向。新语料 covariance 只用于测量旧方向能量和构造匹配随机对照。

对任意旧方向集合 \(Q\)，计算每层在新语料中的方差能量：

\[
E_l(Q)
=
\frac{
\operatorname{tr}
\left(
Q^\top
\Sigma_{l,\mathrm{new}}
Q
\right)
}{
\operatorname{tr}
\left(
\Sigma_{l,\mathrm{new}}
\right)
}.
\]

同时记录新语料均值在该空间中的能量：

\[
M_l(Q)
=
\lVert
Q^\top\mu_{l,\mathrm{new}}
\rVert_2^2.
\]

## 11. 随机对照

随机对照至少匹配：

1. 删除维数；
2. 新语料中被删除的 activation variance；
3. 作用层范围；
4. 相同投影公式；
5. 相同输入文本和 batch 顺序。

对实验 A 生成 rank-319 随机 drop spaces；对实验 B 生成 rank-320 随机 drop spaces。

步骤：

1. 使用固定候选 seed 列表生成至少 1000 个 Haar 随机正交子空间；
2. 使用新语料 baseline covariance 离线计算每个候选的层级能量；
3. 定义跨层能量匹配目标；
4. 为 A 和 B 各选择能量误差最小的 3 个随机子空间；
5. 在查看反事实输出前冻结 seed 和 projector hash。

能量匹配目标建议为：

\[
\operatorname{energy\_mismatch}(Q)
=
\frac1{12}
\sum_{l=1}^{12}
\left[
\log
\frac{
E_l(Q)+\epsilon
}{
E_l(Q_{\mathrm{target}})+\epsilon
}
\right]^2.
\]

同时保留一个纯维数匹配但不做能量筛选的随机 seed，用于判断能量匹配是否改变结论。

## 12. 主投影定义

主分析使用完全删除目标方向的 through-origin intervention：

\[
O'_l
=
O_l
-
(O_lQ_{\mathrm{drop}})
Q_{\mathrm{drop}}^\top.
\]

这与 zero-ablation 端点一致，适合作为因果消融。

均值敏感性只在 Pilot 主结果通过后执行：

\[
O'_l
=
O_l
-
\left[
(O_l-\mu_{l,\mathrm{new}})
Q_{\mathrm{drop}}
\right]
Q_{\mathrm{drop}}^\top.
\]

均值敏感性不得作为主结果替代 through-origin intervention。

## 13. Pilot 条件

Pilot 使用相同的 256 份文本、tokenization、batch 顺序和 baseline。

### 13.1 主条件

| 条件 | 投影范围 | 说明 |
|---|---|---|
| original | 无 | 原始模型基线 |
| zero_all | Layer 1–12 | 12 层 attention output 全部清零 |
| A_all | Layer 1–12 | 保留旧 K64-low-449 |
| B_all | Layer 1–12 | 删除旧 K640-high-320 |
| random_all | Layer 1–12 | 删除维数和能量匹配的随机空间 |
| A_L12 | Layer 12 | 仅最后一层应用 A |
| B_L12 | Layer 12 | 仅最后一层应用 B |
| random_L12 | Layer 12 | 仅最后一层应用随机对照 |

Original 只运行一次。

Pilot 第一轮只使用一个最优能量匹配随机 seed。只有目标方向相对随机方向显示稳定差异时，才补充其余两个随机 seed。

### 13.2 为什么保留 Layer-12-only

旧语料中，所有层同时零消融距离几乎等于 Layer 12 单独零消融距离。新语料必须重新检查该现象。

计算：

\[
D_{\mathrm{incremental}}
=
D_{\mathrm{all}}
-
D_{\mathrm{L12}}.
\]

如果 all-layer 与 Layer-12-only 几乎相同，则全层结论主要由最后一层支配，不应立即扩展为 12 个单层条件。

## 14. 推理成本控制

实验 A 不应直接用 449 维 keep basis 计算，应使用 319 维 drop complement：

\[
O'_l
=
O_l
-
(O_lQ_{319})
Q_{319}^{\top}.
\]

实验 B 使用 320 维 drop basis：

\[
O'_l
=
O_l
-
(O_lQ_{320})
Q_{320}^{\top}.
\]

每次投影使用两次低秩乘法，不显式构造并乘以完整 \(768\times768\) projector。

同一 run 内：

- 模型只加载一次；
- tokenization 只执行一次；
- baseline 只执行一次；
- 所有条件复用相同 encoded inputs；
- 使用 inference mode；
- hook 必须 exception-safe；
- 每个条件结束后验证 hook 已移除；
- 及时释放不再使用的 GPU tensor。

如果 A 与 B 的保留空间高度重合，可只对 A 使用 3 个随机 seeds，B 保留一个 matched control 作为敏感性，从而减少推理次数。

## 15. 数值精度审计

新语料输入长度、字符和 token 分布可能与旧语料显著不同。Pilot 前先用至少 32–64 份文本验证：

- FP16 与 FP32 baseline CLS agreement；
- FP16 与 FP32 intervention distance agreement；
- frozen-head logits agreement；
- 非有限值；
- projection checksum；
- hook identity。

预设建议：

- CLS/logit prediction Pearson 不低于 0.999；
- condition mean distance 相对差异不超过 1%；
- 不允许 NaN/Inf；
- FP16 不通过时使用 FP32，而不是改变方向或样本。

精度规则必须在查看主要条件差异前冻结。

## 16. 每份文本的输出

每个条件、每份文本保存：

    condition
    scope
    random_seed
    text_id
    text_sha256
    token_count
    cls_squared_l2_distance
    cls_cosine_similarity
    baseline_cls_norm
    modified_cls_norm
    frozen_head_logit_mae
    frozen_head_logit_max_abs
    finite

如果冻结头输出为单值或二分类 logits，应明确实际 shape 和比较定义。

## 17. 汇总指标

主要指标：

\[
D_{\mathrm{CLS}}
=
\lVert
h_{\mathrm{condition}}
-
h_{\mathrm{original}}
\rVert_2^2.
\]

报告：

- mean/median CLS squared L2 distance；
- CLS cosine similarity；
- frozen-head logit MAE；
- frozen-head output Pearson/Spearman；
- hidden norm shift；
- 非有限值数量；
- 相对 zero-all 的 representation recovery；
- report-level paired bootstrap 95% CI。

主要对比：

\[
\Delta_A
=
D_A-D_{\mathrm{random,A}},
\]

\[
\Delta_B
=
D_B-D_{\mathrm{random,B}},
\]

\[
\Delta_{\mathrm{scope}}
=
D_{\mathrm{all}}-D_{\mathrm{L12}}.
\]

所有 bootstrap 使用相同 report indices 做 paired resampling。保留未四舍五入的底层数值。

## 18. Pilot 继续条件

只有满足以下条件才进入 Confirmation：

1. 目标方向与能量匹配随机方向的 paired-bootstrap 95% CI 不包含 0；
2. 效果方向在 token-length 四分位中一致；
3. 若有 source/category，主要类别方向一致；
4. 效果不完全由 Layer 12 单独解释，或明确将结论限制为 Layer 12；
5. FP16/FP32 数值审计通过；
6. 无非有限输出；
7. 方向、样本、随机 seeds 和代码 hashes 完整。

如果 A 和 B 结果几乎相同且方向高度重合，Confirmation 只保留一个主条件，另一个作为一次性敏感性检查。

如果目标方向与随机方向无稳定差异，应停止该迁移消融，不增加样本量寻找显著性。

## 19. Confirmation

Confirmation 使用与 Pilot 完全不重叠的 1024 份文本。

在运行前冻结：

- 旧方向 artifact；
- 主实验 A 或 B；
- 保留的敏感性条件；
- 三个随机 seeds；
- layer scope；
- projection semantics；
- 评价指标；
- bootstrap 规则；
- 决策标准。

Confirmation 不允许根据结果重新选择 coverage 阈值、方向数量、layer scope 或随机对照。

## 20. 实现位置

建议新增：

    src/activation_direction_transfer.py
    configs/activation_direction_transfer.yaml
    tests/test_activation_direction_transfer.py

Notebook 仅调用模块、展示 canonical 表格和图，不在 cell 中定义核心业务逻辑。

可新增独立 notebook：

    notebooks/activation_direction_transfer_pipeline.ipynb

不要继续把迁移实验追加到旧语料 activation-rank notebook，以避免新旧语料输出和执行状态混杂。

## 21. 输出目录

建议：

    artifacts/checkpoint_activation_rank/transfer/
    └── old_financial_to_<new_corpus>_v1/
        ├── old_direction_sets.npz
        ├── old_direction_manifest.json
        ├── pilot_evaluation_manifest.parquet
        ├── confirmation_evaluation_manifest.parquet
        ├── new_corpus_baseline_moments.npz
        ├── new_corpus_direction_energy.parquet
        ├── random_control_manifest.parquet
        ├── report_level_distances.parquet
        ├── condition_summary.parquet
        ├── precision_audit.parquet
        ├── transfer_results.png
        └── manifest.json

Pilot 和 Confirmation 应写入独立子目录，避免在确认阶段覆盖 Pilot。

## 22. Manifest

最终 manifest 至少记录：

- 旧语料 analysis/coverage manifests 和 hashes；
- 旧方向 artifact hash；
- 新语料文件和文本内容 hash；
- 新语料 sample manifest hash；
- checkpoint/tokenizer identity；
- max length 和 truncation；
- direction selection K、阈值和维度；
- scope；
- random candidate pool 和 selected seeds；
- dimension/energy matching误差；
- projection semantics；
- compute dtype；
- physical GPU 和 process-local CUDA mapping；
- code/config hashes；
- 所有输出文件 hashes；
- labels_or_returns_loaded=false；
- scope=cross_corpus_fixed_direction_functional_transfer。

## 23. 必须添加的测试

至少包括：

1. 旧方向数量和阈值选择；
2. Q 的正交性；
3. projector 的对称性和幂等性；
4. A 的 keep-space 与 drop-complement 等价；
5. B 的 mask 公式正确；
6. R=0 identity；
7. R=768 zero；
8. hook 异常后必然移除；
9. 多条件之间无残留 hook；
10. evaluation sample ID/text-hash 不重叠；
11. 旧 PCA sample 与新 evaluation sample 重叠审计；
12. random direction deterministic；
13. dimension matching；
14. energy-matching objective；
15. report-level paired bootstrap；
16. baseline 只运行一次；
17. manifest/hash 验证；
18. 不加载标签、收益率、split 或 exposure；
19. synthetic Run All。

## 24. 允许的结论

若目标方向相对能量和维数匹配的随机方向具有稳定额外影响，可以表述为：

> 旧金融研报语料中识别出的特定 attention coverage directions，在新语料上表现出可迁移的功能重要性。

若效果主要来自 Layer 12，应限制为：

> 该迁移效应主要由最终 Transformer block 的 attention output 驱动。

若旧方向与随机方向无差异，应表述为：

> 未发现旧语料 coverage directions 在新语料上具有超出维数和激活能量的特殊功能影响。

无论结果如何，本实验均不能直接声称：

- 新语料具有相同 coverage geometry；
- 旧方向是新语料内生高/低 coverage directions；
- 方向具有金融 alpha；
- 结果构成因果或经济预测证据。

## 25. 后续阶段

若固定旧方向在新语料上显示稳定迁移效应，再进入方案二：

1. 为新语料建立独立 activation-rank run；
2. 重新估计新语料 moments、SVD 和 coverage directions；
3. 比较 old-derived 与 new-derived subspaces；
4. 分别运行迁移方向和内生方向消融；
5. 只有在严格 walk-forward train-only 变换下，才进入增量 alpha probe。
