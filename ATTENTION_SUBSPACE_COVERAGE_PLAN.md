# Attention Output 高方差子空间累计覆盖计算计划

## 1. 任务定位

本任务是现有 activation-rank 实验的无标签几何扩展。它不重新运行模型，不读取标签、收益率、行业暴露或时间切分，仅复用已经生成的 attention output 特征子空间。

目标是计算 Layer 1 至 Layer 12 attention output 高方差写入子空间的：

- 层间重合程度；
- 随层数增加的累计覆盖；
- 全层共同较少写入的低覆盖方向。

这里衡量的是 residual 坐标空间中的 attention write-space coverage。结果不应解释为完整的信息处理、attention 读取范围或严格的跨层因果传播。

## 2. 输入

读取现有 canonical analysis 产物：

    artifacts/checkpoint_activation_rank/runs/financial_reports_v2/
    └── analysis/
        ├── subspaces.npz
        ├── rank_metrics.parquet
        └── manifest.json

每层使用 ordinary-token、自然分布、过滤异常值后的 attention output 特征向量和特征值：

    token_natural_filtered__attention_output_01__eigenvectors
    ...
    token_natural_filtered__attention_output_12__eigenvectors

以及相应的：

    token_natural_filtered__attention_output_XX__eigenvalues

执行前必须验证上游 analysis manifest 及文件哈希。不得从不匹配的 run directory、stream 或 activation site 混用子空间。

## 3. K 的定义

每层 attention output 的隐藏维度为 768。其中心化协方差特征向量按特征值从大到小排列：

\[
V_l=[v_{l,1},v_{l,2},\ldots,v_{l,768}],
\qquad
\lambda_{l,1}\ge\lambda_{l,2}\ge\cdots\ge\lambda_{l,768}.
\]

K 表示每层取前多少个高方差方向：

\[
V_{l,K}\in\mathbb R^{768\times K}.
\]

预设：

    K_GRID = [64, 128, 256, 384, 512, 640]
    PRIMARY_K = 640

PRIMARY_K=640 与现有联合恢复实验的 99.7% CLS 表征恢复结果对应。K 网格用于判断结论是否依赖人为阈值。不得在查看标签或收益结果后重新选择 K。

每层都有自己的 \(V_{l,K}\)。不同层的第一个 component 并不是同一个方向。

## 4. 数值预检

对每个 layer 和 K，取：

\[
V_{l,K}=V_l[:,1:K].
\]

构造 projector 前验证：

\[
V_{l,K}^{\top}V_{l,K}\approx I_K.
\]

至少记录：

- 最大正交误差；
- 输入 dtype；
- eigenvector/eigenvalue shape；
- eigenvalue 是否非负且单调不增；
- 上游 stream、site、layer 和 run fingerprint。

## 5. 单层高方差子空间

对每层构造正交投影矩阵：

\[
P_{l,K}=V_{l,K}V_{l,K}^{\top}.
\]

必须验证：

\[
P_{l,K}=P_{l,K}^{\top},
\qquad
P_{l,K}^{2}\approx P_{l,K},
\qquad
\operatorname{tr}(P_{l,K})\approx K.
\]

Projector 只表示该层 attention output 的 top-K 写入方向，不代表该层只读取这些输入方向。

## 6. 两层之间的重合度

对每对层 \(i,j\) 计算：

\[
\operatorname{overlap}_{ij,K}
=
\frac{\operatorname{tr}(P_{i,K}P_{j,K})}{K}
=
\frac{\lVert V_{i,K}^{\top}V_{j,K}\rVert_F^2}{K}.
\]

解释：

- overlap=1：两个 K 维子空间完全相同；
- overlap=0：两个子空间完全正交；
- 中间值：部分重合。

同时计算 principal angles：

    cross = V_i.T @ V_j
    singular_values = svd(cross, compute_uv=False)
    principal_angle_cos2 = singular_values ** 2

为每对层保存：

- overlap；
- mean/min/max cos²；
- principal-angle cos² 的分位数；
- layer distance \( |i-j| \)。

### 6.1 随机子空间基准

两个随机 K 维子空间在 768 维空间中的期望归一化 overlap 约为：

\[
E[\operatorname{overlap}_{ij,K}]=\frac{K}{768}.
\]

因此必须同时报告：

\[
\operatorname{excess\_overlap}_{ij,K}
=
\operatorname{overlap}_{ij,K}-\frac{K}{768}.
\]

参考随机基准：

- K=64：0.0833；
- K=128：0.1667；
- K=256：0.3333；
- K=384：0.5000；
- K=512：0.6667；
- K=640：0.8333。

不能把 K=640 时 overlap=0.85 直接解释为高度重合，因为它仅略高于随机基准。

## 7. 全层累计覆盖矩阵

对 12 层 projector 求平均：

\[
C_K=\frac{1}{12}\sum_{l=1}^{12}P_{l,K}.
\]

特征分解：

\[
C_K=Q_K\Gamma_KQ_K^{\top}.
\]

其中：

\[
\Gamma_K=\operatorname{diag}(\gamma_1,\ldots,\gamma_{768}),
\qquad
0\le\gamma_r\le1.
\]

对任意单位方向 \(q\)：

\[
q^{\top}C_Kq
=
\frac{1}{12}\sum_{l=1}^{12}\lVert V_{l,K}^{\top}q\rVert_2^2.
\]

因此：

- \(\gamma_r\approx1\)：该方向被几乎所有层的 top-K 子空间覆盖；
- \(\gamma_r\approx0.5\)：该方向具有中等跨层覆盖；
- \(\gamma_r\approx0\)：该方向很少被 12 层 attention top-K 写入空间覆盖。

必须验证：

\[
C_K=C_K^{\top},
\qquad
C_K\succeq0,
\qquad
\operatorname{tr}(C_K)\approx K.
\]

## 8. 逐层累计曲线

对前 \(m=1,\ldots,12\) 层分别计算：

\[
C_{K,m}=\frac{1}{m}\sum_{l=1}^{m}P_{l,K}.
\]

对每个 \(K,m\) 输出：

- effective_coverage_rank；
- directions_coverage_below_005；
- directions_coverage_below_010；
- directions_coverage_above_050；
- directions_coverage_above_090；
- 最小、最大和分位数 coverage eigenvalues。

Effective coverage rank 使用 \(C_{K,m}\) 的非负特征值归一化后计算 entropy effective rank：

\[
p_r=\frac{\gamma_r}{\sum_s\gamma_s},
\qquad
\operatorname{erank}(C_{K,m})
=
\exp\left(-\sum_{r:p_r>0}p_r\log p_r\right).
\]

解释：

- 如果各层方向高度重合，effective coverage rank 随 \(m\) 增长很慢；
- 如果各层子空间不断旋转，effective coverage rank 会随 \(m\) 增长并趋近 768；
- 如果只有少数方向跨层重复，coverage spectrum 会同时出现高覆盖头部和低覆盖尾部。

## 9. 提取全层低覆盖方向

在 PRIMARY_K=640 下，将 \(C_K\) 的特征值按从小到大排列，并保存对应特征向量：

    low_coverage_vectors = eigenvectors[:, :128]
    low_coverage_eigenvalues = eigenvalues[:128]

同时保存 top coverage directions，便于后续做对照：

    high_coverage_vectors = eigenvectors[:, -128:]
    high_coverage_eigenvalues = eigenvalues[-128:]

这些方向只作为未来 alpha probe 的预注册候选。本阶段不得读取标签、收益率或根据经济结果改变方向数量。

## 10. 建议伪代码

    for k in K_GRID:
        bases = []
        projectors = []

        for layer in range(1, 13):
            V = load_eigenvectors(layer)[:, :k]
            assert_allclose(V.T @ V, eye(k), atol=ORTHOGONAL_TOL)

            P = V @ V.T
            assert_allclose(P, P.T, atol=SYMMETRY_TOL)
            assert_allclose(P @ P, P, atol=PROJECTOR_TOL)

            bases.append(V)
            projectors.append(P)

        # Pairwise overlap and principal angles.
        for i in range(12):
            for j in range(12):
                cross = bases[i].T @ bases[j]
                singular = svd(cross, compute_uv=False)
                overlap = sum(singular ** 2) / k
                random_baseline = k / 768
                excess_overlap = overlap - random_baseline

        # Prefix cumulative coverage.
        for m in range(1, 13):
            C_m = sum(projectors[:m]) / m
            eigenvalues = eigvalsh(C_m)
            compute_coverage_metrics(eigenvalues)

        # All-layer coverage.
        C = sum(projectors) / 12
        eigenvalues, eigenvectors = eigh(C)
        assert_allclose(trace(C), k, atol=TRACE_TOL)

## 11. 实现位置

新增：

    src/activation_subspace_coverage.py
    configs/activation_subspace_coverage.yaml
    tests/test_activation_subspace_coverage.py

Notebook 只负责显式调用、展示表格和图，不在 notebook cell 内定义核心业务逻辑。

建议在现有 activation-rank notebook 增加一个独立 cell：

    notebooks/activation_rank_pipeline.ipynb

该 cell 只能调用模块函数并读取 canonical 输出，不得重新实现矩阵计算。

## 12. 输出目录和文件

输出到独立扩展目录：

    artifacts/checkpoint_activation_rank/runs/financial_reports_v2/
    └── extensions/
        └── attention_subspace_coverage_v1/
            ├── pairwise_overlap.parquet
            ├── cumulative_coverage.parquet
            ├── coverage_eigenvalues.parquet
            ├── coverage_summary.parquet
            ├── low_coverage_subspace.npz
            ├── pairwise_overlap_heatmap.png
            ├── cumulative_coverage.png
            ├── coverage_spectrum.png
            └── manifest.json

### 12.1 pairwise_overlap.parquet

至少包含：

    k
    layer_i
    layer_j
    layer_distance
    overlap
    random_overlap_baseline
    excess_overlap
    principal_cos2_mean
    principal_cos2_min
    principal_cos2_max

### 12.2 cumulative_coverage.parquet

至少包含：

    k
    prefix_layers
    effective_coverage_rank
    directions_coverage_below_005
    directions_coverage_below_010
    directions_coverage_above_050
    directions_coverage_above_090

### 12.3 coverage_eigenvalues.parquet

至少包含：

    k
    component
    coverage_eigenvalue
    cumulative_share

### 12.4 manifest.json

至少记录：

- upstream analysis manifest SHA-256；
- subspaces.npz 和 rank_metrics.parquet SHA-256；
- stream、site kind、layers；
- K grid 和 primary K；
- 随机子空间基准定义；
- 数值容差；
- code/config SHA-256；
- 所有输出文件 SHA-256；
- labels_or_returns_loaded=false；
- scope=descriptive_attention_write_space_geometry。

## 13. 图表

生成三张主图：

1. Pairwise overlap heatmap：
   - 每个 K 一张或使用 small multiples；
   - 主图优先显示 excess overlap，而不是 raw overlap；
   - 对角线单独标记，不用于总结层间重合。

2. Cumulative coverage：
   - x 轴为累计层数 \(m\)；
   - y 轴为 effective coverage rank / 768；
   - 每条线代表一个 K。

3. Coverage spectrum：
   - x 轴为 coverage component；
   - y 轴为 \(C_K\) eigenvalue；
   - 标注 0.05、0.50、0.90 阈值；
   - PRIMARY_K=640 作为主面板，其他 K 作为敏感性分析。

## 14. 必须添加的合成测试

### 14.1 完全相同子空间

- pairwise overlap=1；
- excess overlap=1-K/768；
- \(C_K\) 有 K 个特征值为 1，其余为 0；
- effective coverage rank=K。

### 14.2 完全正交子空间

- pairwise overlap=0；
- 累计覆盖维数随层数增长；
- 在总维度允许时，联合覆盖维数等于各子空间维数之和。

### 14.3 随机旋转子空间

- projector 对子空间内部正交旋转保持不变；
- overlap、coverage spectrum 和 low-coverage basis span 保持不变。

### 14.4 数值不变量

- \(C=C^{\top}\)；
- \(C\) 半正定；
- 所有 coverage eigenvalues 位于数值容差下的 [0,1]；
- \(\operatorname{tr}(C)\approx K\)；
- projector 幂等；
- 结果对层读取顺序确定且可复现。

### 14.5 已知随机基准

用多组固定随机种子生成随机 K 维子空间，验证平均 overlap 接近 \(K/768\)。该测试使用宽松统计容差，不依赖单次随机结果。

## 15. 验收标准

实现完成必须满足：

- 不触发模型加载或 GPU 推理；
- 不读取标签、收益率、split 或 exposure 数据；
- 所有上游输入经过 manifest/hash 验证；
- 所有 K 和 12 层均有完整输出；
- 数值不变量全部通过；
- 随机 overlap 基准被显式报告；
- 低覆盖 basis 保存并可重新加载；
- 输出目录包含 hash-linked manifest；
- targeted tests 全部通过；
- notebook 仅展示 canonical 产物。

## 16. 解释边界

最终报告必须明确：

- 这是 attention output 写入方向的几何覆盖；
- 不代表 attention 读取了哪些输入信息；
- 不包含 MLP 对 residual stream 的处理；
- residual 路径会保留未被 attention 更新的方向；
- 不进行 Jacobian transport，因此不是严格的跨层因果信息流；
- 直接比较不同层 projector 隐含它们共享同一 768 维 residual 坐标表示；
- 低 coverage 不能直接称为未处理信息；
- 低覆盖方向只能作为后续增量 alpha 的候选；
- 当前分析是全局无标签描述，不能直接作为正式 OOS alpha 的 train-only PCA basis。

## 17. 后续 alpha 阶段

本任务完成后，若 coverage spectrum 显示稳定的低覆盖方向，可在独立、严格 walk-forward 的 alpha 实验中：

1. 在每个 fold 的训练历史内重新估计所需的子空间或固定预注册的 coverage 定义；
2. 将报告级表征投影到 low-coverage 和 high-coverage 子空间；
3. 分别拟合 continuous-label Ridge；
4. 检验 low-coverage prediction 在 Layer 12、冻结头和 n_texts 之外的 incremental Rank IC；
5. 保持 2023 final test 关闭，直到候选、方向和决策标准被锁定。

不得直接用本全时期 activation-rank basis 产生正式 OOS alpha 结论。
