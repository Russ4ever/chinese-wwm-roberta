# Current Layer Probe experiment status

Last updated: 2026-09-01 (Asia/Shanghai)

This file is the handoff source for the active full-history continuous-label
Layer Probe run. Read it before modifying configs, rerunning a stage, or
interpreting results. Separate confirmed repository state from remote execution
state that has only been reported through the running notebook.

## Repository and active entry point

- Repository: `https://github.com/Russ4ever/chinese-wwm-roberta`
- Branch at the time of this update: `main`
- Repository HEAD before the current uncommitted activation-rank fix: `56eaf9e`
  (`headcheck`), also observed at `origin/main`.
- Mac checkout: `/Users/fanjingqi/Documents/ChatGPT/Chinese-wwm-roberta`
- Remote-server checkout: `/home/intern_fjq_2026/Projects/chinese-wwm-roberta`
- Active notebook on the server:
  `/home/intern_fjq_2026/Projects/chinese-wwm-roberta/notebooks/layer_probe_walk_forward_pipeline.ipynb`
- Active configs:
  - `configs/layer_probe_walk_forward.yaml`
  - `configs/probe_dataset_walk_forward.yaml`
- The historical notebook `notebooks/layer_probe_pipeline.ipynb` and its Cell 2
  paths must not be casually changed. The walk-forward work uses a new notebook
  and isolated output directories.

## Primary protocol

- Main labels remain the continuous targets `residual_signed_raw` and
  `delta_log_dispersion`; binary sentiment is disabled and non-blocking.
- The target bundle deliberately has no static split:
  `configs/probe_dataset_walk_forward.yaml` sets `splits.enabled: false`.
- Walk-forward training starts from history dated 2014-01-01. Validation uses
  annual expanding folds covering 2018-01-01 through 2022-12-31. A fold trains
  only on earlier features whose labels were knowable before that fold.
- Validation model selection may use only labels available by 2022-12-31.
- The 2023 final test is closed. `RUN_FINAL_TEST` and
  `strict_test.open_final_test` must remain false until candidates, alpha rules,
  expected directions, and decision criteria are preregistered.

## Static FY0 CSI300 conflicting alternative

- On 2026-08-24 a separate static alternative was implemented at:
  - `configs/layer_probe_static_fy0_csi300.yaml`
  - `notebooks/layer_probe_static_fy0_csi300_pipeline.ipynb`
  - `src/layer_probe_static.py`
- This does **not** replace or relax the main walk-forward protocol above.  It
  deliberately ignores `label_available_date` for continuous-label split
  membership, opens validation and test in one Run All, and writes only under
  `artifacts/continuous_label_layer_probe/runs/static_fy0_csi300_2025h1_v1`.
  Its empirical results will not be comparable to the closed/PIT mainline.
- Fixed feature windows are train 2014-01-01 through 2022-12-31, validation
  2023-01-01 through 2024-12-31, and test 2025-01-01 through 2025-08-01.
- The only continuous tasks are residual FY0 and 1/3-month fixed/market FY0
  dispersion.  FY1/FY2 and active-panel tasks are excluded.  Ridge modeling is
  restricted to Layer 1 through Layer 12; Layer 0 remains descriptive only.
- Both continuous-label and direct-return Ridge are fit at report level.  The
  fixed report predictions are then averaged by `symbol x trading_date`.
  Historical CSI300 membership is used only as the final validation/test RankIC
  evaluation mask and never filters fitting or alpha selection.
- Direct-return labels alone retain a five-trading-day split-boundary purge.
  Multiple reports for one stock-day receive inverse-count weights so the
  stock-day training weight sums to one.
- The accelerated implementation scans each representation layer once for all
  six factor sources, reuses weighted primal sufficient statistics across the
  alpha grid, validates the Torch solver against sklearn, and defaults to
  physical GPU 1.  It writes wide Layer 1-12 predictions rather than a
  duplicated long prediction table.
- A real-data remote notebook run was reported on 2026-08-24.  It reached the
  report-level Ridge stage (notebook section 5) and failed in the preliminary
  Torch-versus-sklearn equivalence audit before the 12-layer fits were written.
  On the first TF32 attempt, `pearsonr` received NaN/Inf after both prediction
  paths had been built and raised instead of allowing the intended non-TF32
  retry.  Because the sklearn fit immediately before it completed, the
  cancellation-prone accelerated moment path is the primary suspect.  The
  remote artifacts have not been synchronized back, so stages 1-4 remain
  user-reported rather than locally verified and the Ridge/evaluation stages
  are not complete.
- The local fix replaces cancellation-prone raw second moments with centered
  weighted online moments, records non-finite equivalence diagnostics, and
  retries with TF32 disabled when the TF32 audit fails.  Plot helpers now close
  returned Matplotlib figures so the notebook's explicit `display(...)` renders
  each figure once.  Local validation after this fix is 79 passing tests,
  including the synthetic Run All pipeline; this is code validation only.
- A second reported section-5 attempt had no non-finite inputs or predictions
  and selected alpha 1000 in both implementations, but both TF32 and FP32
  missed the deliberately strict equivalence thresholds.  The FP32 attempt had
  minimum prediction Pearson 0.9999064579 and maximum validation-Spearman
  difference 0.0005920301, versus required 0.99999 and 0.0001.  The local code
  now uses user-approved tolerances of minimum Pearson 0.9999 and maximum
  validation-Spearman difference 0.001.  Under the reported values TF32 remains
  rejected while FP32 passes with the same selected alpha; FP64 remains a final
  audited fallback only.  Solver-audit tolerances are normalized out of the
  upstream artifact hash, preserving compatibility with the existing
  representation/fixed-head/target artifacts, while the Ridge output records a
  separate numerical-policy fingerprint.  This revision has passed local tests
  but has not yet been confirmed on the remote real-data run.
- The completed remote static notebook was subsequently synchronized to this
  checkout and locally inspected.  It has execution counts 1 through 11, no
  error output, and final status `completed`.  Section 5 reports 939,831 OOS
  report-level Ridge predictions for six tasks across Layers 1 through 12;
  Section 6 reports 137,441 CSI300 `symbol x trading_date` factor rows, 144
  Rank-IC summary rows, and 936 layer-correlation rows.  The generated remote
  manifests and full CSV/Parquet tables have not yet been synchronized, so the
  notebook verifies stage completion and headline counts but not all row-level
  values or artifact hashes.
- A separate opened-test exploration notebook now exists at
  `notebooks/layer_probe_residual_dense_test_pipeline.ipynb`.  It does not
  refit Ridge or change the static protocol.  It reads the existing residual
  FY0 stock-day predictions, carries Layers 3/6/9/11/12 under eight explicitly
  exploratory event/5-day/20-day/60-day/until-next-report rules, activates
  tradable signals on the next trading session, and writes only under
  `dense_residual_test_exploration_v1` inside the static run directory.  Local
  notebook structure and synthetic no-look-ahead tests pass, but no real-data
  remote execution of this densification notebook has yet been confirmed.

## Label-free checkpoint activation-rank extension

- The independent activation-rank extension is implemented at
  `configs/activation_rank.yaml`, `notebooks/activation_rank_pipeline.ipynb`,
  `src/activation_rank.py`, and `tests/test_activation_rank.py`. It does not read
  label, return, exposure, or split artifacts and remains separate from the
  continuous-label Layer Probe mainline.
- One frozen-backbone forward captures 49 sites: residual streams 0 through 12
  and, for each Transformer block, concatenated attention-head output `Z`, the
  pre-residual attention output projection, and the pre-residual MLP write.
  Ordinary tokens exclude tokenizer-marked special/PAD positions; CLS is
  accumulated separately. Attention matrices and raw token activations are not
  stored.
- The v1 remote attempt completed notebook cells 1-4 and the eight 5-million-
  token shard targets. Its 502,928-token pilot used 1,410 reports, selected batch
  size 512 and FP16, rejected BF16, and found that FP16 met the spectral
  agreement criteria against FP32. Cell 5 then stopped at the pooled norm audit:
  maximum mean shift 0.0105536, relative norm-standard-deviation shift 21531.89,
  and filtered fraction 0.317864. Cells 6-8 did not run, and no valid v1 rank
  conclusion exists.
- The v1 failure was a protocol defect rather than evidence of activation drift
  or dimensional collapse. The pilot had saved FP32 5-sigma norm thresholds
  while the main scan used FP16; near-constant LayerNorm norms were quantized
  differently. The old standard-deviation ratio also divided by an almost-zero
  FP32 standard deviation, and the 1,410-sample CLS pilot was too small for a
  universal blocking 1% standard-deviation gate.
- On 2026-08-25 the protocol was corrected and versioned as
  `checkpoint_activation_rank_v2.0`, with default run directory
  `artifacts/checkpoint_activation_rank/runs/financial_reports_v2`. FP32 remains
  the spectral-accuracy reference, but 5-sigma norm statistics and thresholds
  now come from the selected main-scan dtype. The selected dtype, threshold
  files, audit policy, and code/config fingerprints all enter the execution
  identity, so incompatible pilot or shard state cannot be reused.
- V2 preserves the v1 hash seed only to keep text ordering and shard assignment
  comparable. Its v2 schema and run directory force a fresh sample manifest,
  pilot, moments, analysis, and mechanism run. V1 files are not required for
  v2 execution; the user plans to remove the failed v1 remote directory
  manually for disk space. The historical failure remains documented here.
- The blocking norm audit now applies to ordinary tokens only. CLS filtering is
  reported as a non-blocking diagnostic, and both filtered and unfiltered CLS
  moments are stored so the effect can be measured directly. Near-zero norm
  standard deviations use a scale floor of
  `compute_dtype_epsilon * abs(pilot_mean)` instead of an unstable near-zero
  denominator. Both the 5-million checkpoint and final stage write a 98-row
  per-population/per-site audit table with exact failures and maxima.
- The notebook now has ten cells. Cells 1-8 remain the idempotent core v2
  pipeline; Cell 9 is a descriptive spectral summary of the canonical analysis
  artifacts. The notebook currently has a mixed execution state: Cell 5 has no
  stored result, while several later cells retain outputs. The canonical remote
  artifacts have not been synchronized and hash-validated locally, so these
  stored outputs are not confirmation of a valid completed v2 run.
- On 2026-09-01 the former Cell 10 was superseded. Its attention and MLP paths
  jointly modified all 12 layers while its residual path modified only Layer 6,
  it evaluated PCA-sample reports, used one mask seed, rounded CE before the
  recovery calculation, clipped small denominators to `1e-8`, skipped an actual
  full-space projection at `N=768`, and hard-coded `cuda:0`. Those results are
  methodologically incomparable and must not be interpreted.
- The replacement Cell 10 is a label-free extension implemented in
  `src/activation_rank_loss_recovery.py` with policy
  `configs/activation_rank_loss_recovery.yaml`. It applies one intervention at a
  time to `attention_output_06`, `mlp_output_06`, or `residual_06`; selects 256
  evaluation reports disjoint from the PCA/rank sample by both `report_id` and
  text hash; reuses identical masks across conditions for three fixed mask
  seeds; retains raw unrounded CE; report-block bootstraps confidence intervals;
  and reports `non_identifiable` rather than a recovery fraction when the zero-
  ablation denominator fails the predeclared magnitude or signal-to-noise gate.
  `N=768` is executed as a real projection and must pass an absolute-loss
  identity audit. Outputs include the held-out manifest, mask audit, per-report
  losses, metrics, site summary, figure, preflight, and a hash-linked manifest.
- Cell 10 uses the configured device (physical GPU 1 by default) and calls the
  standard preflight. Its loss is explicitly named an MLM proxy because it
  combines the fine-tuned backbone with the base model's MLM head; it is not the
  checkpoint's original supervised loss and does not establish downstream
  return predictiveness. No real execution of the replacement Cell 10 has been
  confirmed.
- Local code validation after the Cell 10 correction is **102 passing tests**.
  This includes deterministic held-out selection, ID/text-hash disjointness,
  multi-seed/same-mask validation, non-identifiable denominator handling,
  full-projection equivalence, exception-safe hook removal, and the ten-cell
  notebook structure. This is code validation only, not empirical completion.

## Confirmed label state

- Full label construction and an outcome-blind coverage audit were completed on
  the server on 2026-08-23.
- Audited source counts:
  - reports: 1,070,181
  - report FY-label rows: 2,625,006
  - confirmation-label rows: 15,750,036
- The audit covers all 21 configured tasks: three residual forecast horizons and
  18 dispersion combinations from 1/3 month, fixed/market/active panel, and
  forecast horizon 0/1/2.
- Canonical server label directory:
  `/home/intern_fjq_2026/Projects/chinese-wwm-roberta/artifacts/report_labels`
- Detailed audited counts and source fingerprints are in
  `audit_reports/continuous_label_audit/server_full_labels_preflight/`.
- The earlier static split audit showed zero validation rows for 3-month
  dispersion and residual tasks because their labels matured after the old
  cutoffs. This motivated the expanding walk-forward implementation; it was not
  evidence that the raw labels were absent.

## Reported remote execution state

The following status was reported from the server notebook but has not yet been
verified by synchronizing its generated manifests back to this checkout.

1. The full-history target-bundle cell was run. On rerun, its initial generic
   warning disappeared because the manifest already existed and the builder was
   skipped.
2. For this intentional unsplit bundle, the expected metadata state is
   `splits == []` and `training_ready == false`. The latter only blocks the old
   static `load_probe_task` loader. The new pipeline reads all valid targets and
   constructs fold membership dynamically with `fold_masks()`.
3. Stage 1 representation extraction was started or was about to run. Its
   completion has **not** been confirmed. Do not claim that representation
   artifacts exist until both the pointer and content-addressed manifest pass
   validation on the server.

Expected server output locations:

- Target bundle:
  `artifacts/probe_dataset_walk_forward_v1`
- Representation store:
  `artifacts/continuous_label_layer_probe/representation_store_full_history_wf_v1`
- Walk-forward run:
  `artifacts/continuous_label_layer_probe/runs/full_history_wf_v1`

Before rerunning Stage 1, check for and validate:

- `artifacts/continuous_label_layer_probe/runs/full_history_wf_v1/representation_pointer.json`
- The `representation_manifest.json` referenced by that pointer
- `representations.npy`, `report_metadata.parquet`, and
  `fixed_head_layer_outputs.parquet` in the referenced directory

Do not infer success from a partially growing hidden temporary directory. Do not
start a second Stage 1 job while the first server kernel is still running.

## Stage 1 operational notes

- Stage 1 processes all 1,070,181 reports, extracts Layer 0 through Layer 12 in
  one backbone forward call per batch, and stores float16 representations. The
  representation array alone is approximately 20 GiB.
- Only physical GPU 1 is selected automatically. GPU 0 activity does not block
  it, but a foreign compute process on physical GPU 1, GPU-1 utilization above
  10%, or GPU-1 memory use above 10% causes a safe stop. The code does not fall
  back to GPU 0.
- With an exact valid representation artifact, Stage 1 reuses it without GPU
  inference. Without one, a first full run may take hours.
- The implementation is correctness- and audit-oriented rather than maximally
  pipelined: it uses an 8-thread shared-server budget, disables tokenizer
  parallelism, and serializes tokenization, GPU inference, CPU transfer, and
  Parquet writing. Low sawtooth GPU utilization can therefore indicate a CPU or
  I/O bottleneck. Do not interrupt an active run merely to optimize it; the
  current temporary output is removed on interruption.
- The cell has little progress output. Verify the active process and physical
  GPU 1 rather than assuming that a quiet cell has stalled.

## Scale handling and pending methodological decision

- Raw CLS representations are intentionally not normalized. Layer-wise norm and
  frozen-head scale differences are part of Scheme A's descriptive diagnostics.
  Scheme A must not be used as a fair layer-ranking probe.
- Walk-forward continuous-label Ridge probes fit a weighted `StandardScaler`
  independently for every task, layer, and fold using training rows only.
  Validation uses the corresponding training scaler.
- Direct-return layer probes also standardize each layer from training data.
- Layer-correlation diagnostics use daily cross-sectional Spearman correlation
  and are insensitive to simple positive rescaling.
- PCA cross-layer factors standardize layer predictions using the historical
  reference sample before fitting PCA.
- **Pending issue:** `layer_consensus`, `layer_disagreement`, and
  `deep_minus_middle` currently combine raw per-layer predictions without first
  matching their reference-sample dispersion. Layers with larger prediction
  variance can dominate these composites. The proposed fix is to fit a
  per-layer mean/scale on the permitted historical reference rows, transform
  validation predictions with those frozen parameters, then construct these
  three composites and record the parameters in the manifest.
- That proposed scale fix has not been implemented as of this update. It does
  not affect Stage 1, Scheme A/A+, Scheme B, direct-return single-layer probes,
  label-factor returns, or layer-correlation outputs. It changes the semantics
  and comparability of Stage 5-6 composite-factor results, so resolve it before
  treating those factors as final candidates.

## Next-task checklist

1. Read this file and inspect `git status --short --branch` before any edit.
2. Ask for or inspect the current server notebook output; do not assume Stage 1
   completed.
3. If Stage 1 completed, validate its pointer, manifest, Layer-12 equivalence,
   shape, file sizes, and hashes before moving on.
4. If Stage 1 is still running, inspect GPU 1, the owning PID, free disk, and the
   temporary artifact growth. Do not launch a duplicate process.
5. Preserve the server project path and existing data paths. In particular, the
   return/Barra paths remain under `/home/intern_fjq_2026/data/` as configured.
6. Resolve and test the Stage 5-6 composite-factor scale decision before final
   validation-factor interpretation.
7. Keep final test closed and do not weaken label cutoffs to recover samples.
8. Before the first real activation-rank run, synchronize the implementation,
   inspect live physical GPU 1/processes/NUMA/CPU affinity/free disk, then run
   notebook cells 1 through 4 and inspect the measured dtype, batch size,
   padding ratio, peak memory, filtering thresholds, and 5m/10m ETA before
   starting cell 5. Do not claim activation-rank completion from local tests.

## Code validation known at handoff

- The walk-forward pipeline was committed in `9974fa2` with targeted tests in
  `tests/test_layer_probe_walk_forward.py`.
- At the earlier implementation handoff, the full local test suite was reported
  as 65 passing tests and the new notebook structure/AST was validated.
- This statement is code validation only, not evidence that the current
  full-data server experiment completed successfully.
- The label-free activation-rank implementation added on 2026-08-25 raises the
  full local suite to 94 passing tests. Its later remote pilot passed, while
  the 5-million-token scan stopped at the flawed pooled norm gate described
  above; no rank/mechanism result is complete or interpretable.
