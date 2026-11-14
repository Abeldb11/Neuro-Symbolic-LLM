# Stage A benchmark (A1, A2, A3)

Compares the plain residual adapter (A1), the auxiliary-loss adapter (A2) and the
FabricPC predictive-coding adapter (A3, K settling steps) on a frozen GPT-2 over three
LexGLUE domains: EUR-Lex, LEDGAR and SCOTUS. A4 (symbolic head) is not part of the runnable
benchmark yet.

## Run it

```bash
pip install -e ".[dev,gpu,visualization]"

# Smoke: seed 0, rank 16, K=4, sequential protocol only
python experiments/run_benchmark.py plan   --output-dir runs/smoke --profile smoke
python experiments/run_benchmark.py train  --output-dir runs/smoke
python experiments/run_benchmark.py eval   --output-dir runs/smoke
python experiments/run_benchmark.py report --output-dir runs/smoke --report-rank 16
```

- `--profile headline` or `full` selects the larger grids. Add `--protocols sequential joint`
  to a smoke plan to also get the sequential-versus-joint gap.
- `train` resumes: finished runs are skipped. Use `--no-resume` to redo them.
- `--eval-config PATH` points the external-metric layer at another settings file.
- `--no-visualization` skips the figures under `report_and_visualization/`.

## Metrics

Results are split in the report into **Internal** (what the adapter does inside the model:
residual compressibility, effective rank, redundancy, gradient conflict, update
commutator, ...) and **External** (how the deployed model behaves).

External metrics, math in `metrics/performance.py`:

| Metric | Report row(s) |
|---|---|
| Validation loss, perplexity (test loss and perplexity at the end) | Validation loss, Perplexity, Held-out test ... |
| Token accuracy and token ECE | Token accuracy (diagnostic), Token calibration ECE |
| Task accuracy, macro-F1, micro-F1, class ECE (label-token probe) | `... label-token probe` rows |
| Forgetting and backward transfer (loss and accuracy) | Validation-loss forgetting, Backward transfer rows |
| Held-out (zero-shot) transfer | Zero-shot gain, Gain on never-trained domains |
| Sequential versus joint gap (budget-matched pairs only) | Sequential minus joint, relative gap |
| Anchor drift on wikitext (KL and loss change) | Anchor KL rows |
| Long-range proxy (position-bucketed loss at 512 tokens) | Long-range proxy |

A cell showing `N/A` was not measured; the reason is stored in `evaluation.json` under
`records[*].deployed_evaluation.external` (`probe`, `anchor`, `long_range_proxy` status
fields). It is never reported as zero.

Sign conventions: backward transfer is negative when the model forgot; the joint gap is
positive when sequential training is worse; anchor KL going up means more drift from the
frozen GPT-2.

## Where things live

| What | File |
|---|---|
| Metric math | `metrics/performance.py` (external), `metrics/representation.py` (internal) |
| Training-time evaluation shared by A1/A2/A3 | `stages/stage_base.py` (`make_eval_fn`, `attach_task_probes`) |
| Post-hoc evaluation and report | `benchmark/core.py` |
| Settings (probe, anchor, long-range) | `configs/benchmark_eval.yaml` |
| Pinned datasets and label specs | `configs/runtime_datasets.yaml` (`legal_task` blocks) |

## Known limitations

- The task probe scores the label name as the next token after the document, with no prompt
  template, and adapters are trained on language modelling only. Expect accuracy near
  chance; use it to compare methods, not as an absolute task score.
- Labels that share a first token cannot be told apart by the probe (check EUR-Lex).
- EUR-Lex is multi-label; the probe predicts one label, so recall is bounded.
- Anchor and long-range corpora must be pinned (`revision`) in `configs/benchmark_eval.yaml`
  for reproducible results.
- The symbolic precision/recall metrics need an A4 retriever and are not measured here.
