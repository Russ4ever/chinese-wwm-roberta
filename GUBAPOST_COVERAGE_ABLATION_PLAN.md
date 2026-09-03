# gubapost 语料覆盖方向消融实验方案

## 1. 研究定位

在 gubapost(股吧帖子)语料的 activation-rank run(`gubapost_v1`)完成后,用该语料**自身**的
attention output coverage directions 做消融实验。这不是跨语料迁移——方向来自新语料自身的 C_K。

核心问题:

> gubapost 语料中,attention output 的低覆盖方向和高覆盖方向,对最终 CLS 表征和冻结头输出
> 有多大功能影响?

## 2. 方向来源

方向从 `gubapost_v1/analysis/subspaces.npz` 重建(纯线性代数,秒级,无 GPU):

- 对每层 l=1..12 取 attention output 的 top-K 特征向量 V_{l,K}
- 构造投影矩阵 P_{l,K} = V_{l,K} V_{l,K}^T
- 全层平均 C_K = (1/12) Σ P_{l,K}
- 特征分解 C_K → 768 个 coverage 特征值 + 特征向量(降序)

阈值切分(已验证 gubapost_v1 实际值):

| K | 阈值 | 方向集 | 维度 |
|---|---|---|---|
| 64 | γ < 0.05 | 低覆盖方向 | **451** |
| 640 | γ ≥ 0.90 | 高覆盖方向 | **318** |

## 3. 四套消融方案

所有干预施加在 12 层 attention output 上(全层同时),使用 through-origin 投影:

    O'_l = O @ Q_keep @ Q_keep^T   (保留 Q_keep 空间)
    O'_l = O − (O @ Q_drop) @ Q_drop^T   (删除 Q_drop 空间 = 保留补空间)

| # | K | 方向集 | 干预 | 保留维度 |
|---|---|---|---|---|
| 1 | 64 | 低覆盖 451 维 (γ<0.05) | **keep** | 451 |
| 2 | 64 | 低覆盖 451 维 | **drop** (= keep 补空间 317) | 317 |
| 3 | 640 | 高覆盖 318 维 (γ≥0.90) | **drop** (= keep 补空间 450) | 450 |
| 4 | 640 | 高覆盖 318 维 | **keep** | 318 |

方案 1 与 2 互补(同一 451 维子空间的 keep vs drop);方案 3 与 4 互补(同一 318 维子空间)。

额外 baseline:

- `original`: 无干预(模型原始输出)
- `zero_all`: 12 层 attention output 全部清零(最大破坏参考)

共 6 个 condition。

## 4. 评估样本

- 256 条文本,与 PCA sample 按 ID + text SHA-256 隔离(零重叠)
- 从 `gubapost_text_sample.parquet`(438K 帖)中选取
- tokenize max_length=512(与 activation-rank run 一致)

## 5. 测量指标(每条文本 × 每个 condition)

| 指标 | 定义 |
|---|---|
| `cls_squared_l2_distance` | ‖CLS_modified − CLS_baseline‖² |
| `cls_cosine_similarity` | cos(CLS_modified, CLS_baseline) |
| `class_1_prob` | 干预后 class_1 概率 |
| `class_1_prob_mae` | |prob_modified − prob_baseline| |
| `class_1_logit` | 干预后 class_1 logit |
| `class_1_logit_mae` | |logit_modified − logit_baseline| |
| `finite` | 无 NaN/Inf |

## 6. 产物(磁盘节省, <5MB)

写入 `gubapost_v1/extensions/coverage_ablation_v1/`:

| 文件 | 内容 |
|---|---|
| `report_distances.parquet` | 256 文本 × 6 condition 逐条指标(~1536 行) |
| `condition_summary.parquet` | 6 行 per-condition 汇总(mean/median) |
| `direction_sets.npz` | 4 组方向矩阵 + coverage 特征值 |
| `manifest.json` | 配置/方向/模型/产物 hash |

## 7. 实现

- Notebook: `notebooks/gubapost_coverage_ablation.ipynb`(自包含,不复用 src 之外的新模块)
- 复用: `SingleSiteProjection`、`resolve_projection_module`、`coverage_spectrum`、`load_layer_bases`、
  `select_disjoint_evaluation_reports`、`_load_candidate`、`_tokenize_evaluation`
- GPU: cuda:1 only, fp16, max_length=512
- 预计耗时 < 1 min(方向计算 ~5s + 模型加载 ~10s + tokenize ~2s + 6 condition 前向 ~30s)

## 8. 可表述的结论

- 方案 1 vs 2: 低覆盖方向是"冗余"还是"必要"?如果 keep 451 维(方案 1)的 CLS 距离接近 zero_all,
  说明这 451 维几乎不携带信息;如果 drop 451 维(方案 2)的 CLS 距离很小,说明删掉它们不影响表征。
- 方案 3 vs 4: 高覆盖方向是"核心写入空间"?如果 drop 318 维(方案 3)造成大距离,说明它们是关键方向;
  如果 keep 318 维(方案 4)恢复大部分表征,说明少量高覆盖方向就够。
- 与 zero_all 对比: 各方案的 CLS 距离相对于 zero_all 的恢复比例。
