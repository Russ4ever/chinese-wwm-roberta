# 实现计划：逐层 CLS 表征恢复曲线（单 GPU cuda:1, batch=256）

## 约束

- 只用 `cuda:1`（不动 cuda:0）
- batch_size: 256（A100 80GB 单卡足够）
- 预计总耗时 ~4 分钟（157 次前向 × 1 batch × ~1.5s）

## 度量替换

| | 旧（MLM CE） | 新（CLS 表征距离） |
|---|---|---|
| 下游指标 | 遮盖 token 的交叉熵 | ‖modified_cls − baseline_cls‖² |
| 模型 | BertForMaskedLM | BinaryClassificationCandidate |
| mask 种子 | 3个随机mask | 不需要 |
| 逐层 | 只 Layer 6 | Layer 1-12 全部 |
| batch_size | 32 | 256 |
| 恢复公式 | (loss_zero − loss_N) / (loss_zero − loss_original) | 1 − dist_N / dist_zero |

## 文件改动

### 1. src/activation_rank_loss_recovery.py（主要重写）

保留不变：`select_disjoint_evaluation_reports`、`SingleSiteProjection`、`validate_loss_recovery_outputs`、`_model_source_fingerprint`

修改：
- `resolve_projection_module`：`layer != 6` → `layer < 1 or layer > 12`
- `load_loss_recovery_policy`：`evaluation.layer` → `evaluation.layers`（列表）+ `evaluation.kinds`，去掉 mask_seeds 校验
- `_load_mlm_proxy` → `_load_candidate`：用 `build_candidate` 替代 BertForMaskedLM
- `_evaluate_condition` → `_evaluate_representation_condition`：前向取 pooled_feature，算 L2 距离 vs baseline_cls
- `_pooled_loss` → `_mean_distance`
- `_bootstrap_losses` → `_bootstrap_distances`
- `summarize_loss_recovery`：去 mask_seed 维度，distance_original=0，recovery=1−dist_N/dist_zero，遍历 layers×kinds
- `_plot_metrics`：12 条曲线（颜色渐变）+ CI 带 + 99% 水平线 + 贡献柱状图
- `run_loss_recovery_stage`：读 layers+kinds，先算基线CLS，循环 layers×kinds×components，manifest 更新 loss_semantics
- `LOSS_SEMANTICS = "cls_representation_distance__per_layer"`

### 2. configs/activation_rank_loss_recovery.yaml

- `evaluation.layer: 6` → `evaluation.layers: [1,...,12]`
- 添加 `evaluation.kinds: ["attention_output"]`
- 去掉 `mask_seeds` / `mask_probability`
- `batch_size: 32` → `256`
- `output.subdirectory` → `extensions/loss_recovery_per_layer_cls_v1`
- schema_version → v1.1

### 3. notebooks/activation_rank_pipeline.ipynb cell 10

更新调用 + 显示 12 条曲线图 + site_summary + 关键N值 metrics

### 4. tests/test_activation_rank_loss_recovery.py

更新 4 个测试适配新接口

### 5. 清理文件

删除：scripts/spectral_loss_recovery.py、scripts/add_notebook_cells.py、_fix_cell2_path_remote.py、_remote_fix_notebook.py

## 执行流程

1. 本地改代码 + config + tests
2. pytest 验证
3. commit → 16 端 git pull
4. 16 上 git rm 清理临时文件
5. nbconvert --execute（cuda:1, batch=256, ~4min）
6. 合并 cell 9-10 输出
7. 16 上 commit notebook
8. 本地 pull sixteen main