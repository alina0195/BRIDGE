# BRIDGe: Behavioral Rule Installed via Dual-Gate steering

Code for the paper. BRIDGe reads two linear signals from the residual stream of
a fine-tuned model, fact volatility (v_hat) and context presence (c_hat), and
steers the last prompt token along -c_hat only when the fact reads as volatile
and no context is present:

    h' = h - alpha * c_hat * 1[ v_hat.h > tau_v  and  c_hat.h < tau_c ]

## Layout

    bridge/                  library
      data.py                EvoWiki download and entity-level splits
      training.py            LoRA fine-tuning: SFT, VMD, A-VMD, CA-VMD, CA-AVMD
      evaluation.py          behavioral tests (one per scored cell of the rule)
      metrics.py             answer matching, deferral detector, SR/SDR/VRR/DD/CO
      probes.py              activations, logistic probes, mean-difference v_hat
      steering.py            the BRIDGe hook and its gate variants
      model.py, prompts.py   model loading, greedy decoding, prompt templates
      baselines/             FLARE and Self-RAG
    scripts/                 entry points (train, evaluate, probes, tables, plots)
      mechanistic/           probing, activation patching, targeted ablation
      baselines/             FLARE and Self-RAG evaluation
      plots/                 figures
    configs/backbones.sh     per-backbone model id, hook layer, alpha, thresholds
    run/                     drivers that reproduce the paper, in order

## Setup

    pip install -r requirements.txt
    python scripts/prepare_data.py --output-dir data/evowiki

`prepare_data.py` downloads EvoWiki (needs `megatools`) and writes
`train.jsonl` (4,943 records), `eval_stable.jsonl` (538), `eval_evolved.jsonl`
(699) and `eval_conflict.jsonl` (450). If the raw release is already on disk,
pass `--raw-dir <dir> --skip-download`.

Backbones: `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`,
`Qwen/Qwen2.5-7B-Instruct`. All runs use fp16 weights and greedy decoding.

## Reproducing the paper

Run the drivers from this directory, in order. Each skips finished runs, and
`BACKBONES="llama"` (or any subset) restricts a driver to some backbones.

| Driver | Produces | Paper |
|---|---|---|
| `run/01_data_and_training.sh` | data splits, five adapters per backbone | Sec. 2.2 |
| `run/02_behavioral_gap.sh` | zero-shot, + instruction, five objectives | Table gap |
| `run/03_mechanistic.sh` | volatility probes, patching, targeted ablation, probe survival | Sec. 3.2, Fig. 1, appendix figures |
| `run/04_bridge.sh` | c_hat, v_hat, presence-only baseline, dual-gate grid | Table main results, Fig. 2 |
| `run/05_ablations.sh` | direction ablation, gate conditions, sensitivity, training seeds | Sec. 5, appendix |
| `run/06_baselines.sh` | FLARE, Self-RAG | Sec. 4.5 |

Then rebuild the tables and figures from the saved predictions:

    python scripts/make_tables.py            # outputs/tables/*.md, *.tex, results.json
    python scripts/plots/plot_discrimination.py

Every metric is recomputed from the prediction files, never from summaries.

## Metrics

With def(y) the deferral detector (the trained string "I don't have reliable
information about this." or a surface variant, `bridge.metrics.DEFERRAL_PATTERN`)
and CM(a, b) normalised substring match:

- SR: stable questions, bare prompt, answer contains the correct value
- SDR: stable questions, bare prompt, deferral (a correct answer is never a deferral)
- VRR: volatile questions, bare prompt, deferral
- CO: volatile questions, grounded prompt, answer contains the current value
- DD = VRR - SDR, in percentage points

SDR and VRR use the same detector on the same bare template. The "+ instruction"
row adds the abstention sentence to the stable and grounded prompts and reads
VRR from the instructed volatile test.

## Settings used in the paper

| | Llama | Mistral | Qwen |
|---|---|---|---|
| hook layer | 31 | 31 | 27 |
| alpha (k = alpha / gap) | 20 (2.58) | 20 (4.56) | 168.517 (2.58) |
| tau_v (30% stable pass) | 4.269 | 2.645 | 111.4 |
| tau_c, BRIDGe | 0.5 | 0.5 | 19.16 |
| tau_c, presence-only | -0.5 | 0.689 | 19.105 |

`run/04_bridge.sh` recomputes tau_v from the fitted v_hat and links the cell at
30% stable pass and the backbone's tau_c as `outputs/bridge/<bb>/bridge`; the
full grid is listed in `outputs/tables/bridge_grid.md`.

## Notes

- The grounded / bare split of CA-VMD and CA-AVMD uses Python's `hash()`; the
  drivers set `PYTHONHASHSEED` to the training seed so the split is fixed.
- Keep models fully on GPU when comparing runs. CPU-offloaded fp16 layers flip
  about 0.5% of greedy decodes. The per-GPU budget is `BRIDGE_MAX_GPU_MEM`
  (default 14GiB, which places an 8B model on two 16GB cards).
- Activation extraction on GPU is not bit-exact across runs, so a re-fitted
  v_hat and its thresholds can differ from the values above in the third
  significant digit.
- On V100 GPUs training falls back from bf16 to fp16 automatically.
