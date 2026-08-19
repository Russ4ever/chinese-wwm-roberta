# 研报一致预期与预期修正标签规范

当前唯一生产实现为 `label_engineering/build_labels.py`（V2.1）。本文件只说明必须保持的口径；
`src/data/data_check/data_quality.ipynb` 仅用于数据质量与描述性复核，不另行定义生产标签。

## 1. 输入与粒度

- 主指标：`FORECAST_NP`。
- 报告级：一篇报告对一个财年的预测是一条记录，保留 `source_report_id` 与稳定的
  `report_signature`。
- 机构日终级：按 `(stock_code, fy, org_id, available_date)` 聚合，同一机构同一市场可用日
  的多条预测取中位数，同时保留冲突计数和取值范围。
- 股票代码始终保存为六位字符串。

## 2. 时间规则

### 2.1 发布时间与市场可用日

原始 `publish_timestamp` / `publish_date` 必须保留。另生成 `available_date`：

- 交易日且发布时间不晚于 14:57:00：归当日；
- 交易日且发布时间晚于 14:57:00：归下一交易日；
- 非交易日：归下一交易日；
- 超出已有交易日历右边界：记为不可对齐，禁止裁剪到最后一个交易日。

交易日历只从项目已有的 CSV/Parquet 文件读取，不在标签流程中抓取或补造。

### 2.2 月频与过期

- 月频快照一律使用当月最后一个交易日，禁止使用自然月最后一天兜底。
- 1/3 个月未来目标使用目标月份的最后一个交易日。
- 预测过期窗口为 180 个自然日：`asof_trading_date - original_publish_date`，不是 180 个交易日。
- 市场信息的纳入、先后顺序和主动更新窗口使用 `available_date`；新鲜度使用原始
  `publish_date`。

## 3. 防泄漏与删失

- 月末快照只使用 `available_date <= asof_month` 的信息。
- 报告级发布前共识只使用严格早于该报告 `available_date` 的其他机构最新预测，并排除本机构。
- 历史与未来覆盖由配置的 `source_window` 判断；不能用文件中实际第一/最后一条研报日期推断，
  因为“没有研报”不等于“数据缺失”。
- 每个正式目标有独立有效标志。目标无效时正式连续值与类别必须为空；`*_raw` 仅用于诊断。

## 4. 数值与阈值

- 相对修正分母使用 `max(abs(base), scale_floor)`，避免负基准反转方向和近零爆炸。
- 盈利、亏损、近零、正负混合分开处理；正盈利到正盈利使用五分类，跨状态使用迁移类别。
- `near_zero`、共识强度和 strong 阈值只允许在训练期内拟合，并按预测财年期限分组。
- 机构等权；共识主统计量为机构最新有效预测的中位数。

## 5. 输出与审计

生产流程输出：

- `clean_report_v2`
- `org_daily_forecast_v2`
- `consensus_snapshot_monthly_v2`
- `revision_label_monthly_v2`
- `report_label_detail_v2`
- `label_metadata_v2.json`

元数据必须记录交易日历路径、14:57 对齐规则、真实标签月末交易日、180 自然日过期口径、
训练期阈值和各覆盖门槛。生成结果不作为源代码提交；应由当前代码和本地数据重建。
