# Aggregate audit reports

This directory is intentionally tracked. It is reserved for small, aggregate,
outcome-blind audit outputs (`.json` and `.md`) produced by repository audit
commands. Do not place raw report text, label values, model arrays, Parquet files,
or other research data here.

Continuous-label coverage audits are written under:

```text
audit_reports/continuous_label_audit/<run-id>/
```

The command refuses to overwrite an existing run directory.

Run from the repository root on the server:

```bash
python label_engineering/audit_continuous_labels.py \
  --config configs/probe_dataset.yaml \
  --run-id server_v3_preflight
```

Sync the resulting `audit_reports/continuous_label_audit/server_v3_preflight/`
directory back with the repository. The command reads the configured label
artifacts but never writes under `artifacts/`.
