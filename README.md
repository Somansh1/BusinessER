# Business Entity Resolution: Amazon ML Challenge 2026

Macro-F0.5 resolution of business entities across three data sources, multilingual, with a 1-to-1 constraint and unseen France at test time.

**Headline (final v4): validation macro-F0.5 0.9862** on 100k held-out S1 (India 0.9823, US 0.9887; t=0.425, alpha=0.7). Blocking ceiling on the same S1 is about 0.9935. Test validator: PASS.

Pipeline: text normalization (two views) -> per-country blocking (TF-IDF kNN, LaBSE exact kNN, key joins) -> hybrid learned cap -> stage-1 LightGBM -> xlm-roberta-large cross-encoder -> stage-2 LightGBM -> 1-to-1 decision. Full write-up: `Documentation_template.md`.

## Repository Layout

```
ber/
├── README.md
├── Documentation_template.md          # Methodology write-up
├── CONTRACT.md                        # Artifact / schema contract between modules
├── requirements.txt                   # Direct dependencies (pinned freeze: env_freeze.txt)
├── src/ber/
│   ├── text.py                        # Two text views, transliteration, junk/leet cleanup, phonetic keys
│   ├── block.py                       # Candidate generation: TF-IDF kNN, LaBSE kNN, key joins, learned cap
│   ├── features.py                    # Pair features, competition ranks, stage-2 features
│   ├── model.py                       # LightGBM, 1-to-1 assignment, threshold tuning, submission writer + validator
│   ├── ce.py                          # Cross-encoder fit / score (xlm-roberta-large)
│   └── run.py                         # Stage CLI
├── pbs/
│   ├── pipeline.pbs                   # Staged pipeline in one GPU job (STAGES env var selects stages)
│   ├── nb.pbs                         # Execute the notebook in place
│   └── zip.pbs                        # Build the challenge submission zip in dist/
├── notebooks/
│   └── entity_resolution.ipynb        # Exploration and analysis
├── working/                           # Artifacts ($BER_WORK), see CONTRACT.md — git-ignored
└── output/                            # matching_results.tsv, candidate_pairs.tsv — git-ignored
```

## Environment Setup

```bash
conda create -n ber python=3.11
conda activate ber
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8 driver
pip install -r requirements.txt
```

Pre-download the two models on an internet-connected node (GPU nodes are offline):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/LaBSE')"
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large'); AutoModel.from_pretrained('FacebookAI/xlm-roberta-large')"
```
GPU jobs set `HF_HUB_OFFLINE=1` (done in `pbs/pipeline.pbs`) and load from the local HF cache.

## Running

Default `STAGES` of `pbs/pipeline.pbs`, in order:
```
prep:train prep:test folds embed:train embed:test block:train blockgpu:train block:test blockgpu:test
features:train ce_data cefit train1 cescore:train train features:test predict1:test cescore:test predict:test
```

Full run (all stages in one job):
```bash
qsub pbs/pipeline.pbs
```

Smoke test on a 20k-S1 real-data sample:
```bash
python -m ber.run sample --n 20000
qsub -v BER_DATA=working/sample,BER_WORK=working/smoke pbs/pipeline.pbs
```

Three-job chain (keeps at most 2 jobs concurrent; each starts only if the previous succeeded):
```bash
J1=$(qsub -v STAGES="prep:train prep:test folds embed:train embed:test block:train blockgpu:train block:test blockgpu:test" pbs/pipeline.pbs)
J2=$(qsub -W depend=afterok:$J1 -v STAGES="features:train ce_data cefit train1 cescore:train train" pbs/pipeline.pbs)
J3=$(qsub -W depend=afterok:$J2 -v STAGES="features:test predict1:test cescore:test predict:test" pbs/pipeline.pbs)
```

Single stage by hand (all stages read/write `$BER_WORK`):
```bash
python -m ber.run prep --split train
python -m ber.run embed --split train
python -m ber.run block --split train          # CPU: TF-IDF kNN + key joins
python -m ber.run block --split train --gpu    # GPU: LaBSE exact top-40
python -m ber.run features --split train
python -m ber.ce train                         # cross-encoder fit
python -m ber.ce score --split test
```
`run.py` stages: `sample prep folds embed block features ce_data train1 train predict1 predict`.
Blocking is append-mode: S1 already present in `{split}_blk_*.parquet` are skipped.

### Environment variables (all optional)

| Variable | Default | Meaning |
|---|---|---|
| `BER_DATA` | `~/amazonml/amazon-ml/dataset` | input TSV directory |
| `BER_WORK` | `ber/working` | artifacts |
| `BER_OUT` | `ber/output` (`$BER_WORK/output` in PBS when `BER_WORK` set) | submission files |
| `BER_EMB` | `sentence-transformers/LaBSE` | embedding model |
| `BER_CE` | `FacebookAI/xlm-roberta-large` | cross-encoder base model |
| `N_VA` | 100000 | val S1 (first of a fixed permutation) |
| `N_CE` | 500000 | cross-encoder fit S1 |
| `FIT_S1` | 350000 | tr S1 per stage-1 half |
| `N_CHUNKS` | 8 | S1 hash chunks for union/features |
| `K_TOTAL` | 60 | learned cap per S1 |
| `K_PER` | 10 | per-method top-k kept in the cap |
| `K_NAME` | 60 | core-name / key kNN depth |
| `LGB_ROUNDS` | 3000 | max boosting rounds |
| `LGB_LR` | 0.05 | LightGBM learning rate |
| `CE_BS` | 64 | cross-encoder batch size (scoring uses 4x) |
| `CE_LR` | 1e-5 | cross-encoder learning rate |
| `CE_MAX` | 6000000 | max cross-encoder fit pairs |
| `CE_BAND` | 0.005 | score pairs with stage-1 p >= this |
| `S2_MIN` | 0.001 | stage 2 sees pairs with p1 >= this |
| `BER_MAXDF` | 0.02 | TF-IDF max document frequency |
| `NJOBS` | `cpu_count()` | CPU worker count |

## Outputs

`$BER_OUT/`:
- `matching_results.tsv` (97 MB on test): 1-to-1 matches, `s1 -> matched ids` (or empty)
- `candidate_pairs.tsv` (3.0 GB on test): every pair the models scored

`$BER_WORK/`: `train_result.json` (t, alpha, val F0.5, per-country/script, importances), `tune_table.csv`, `blocking_report.json`, `lgb0.txt`, `lgb1_{0,1}.txt`, `lgb2.txt`, `ce_model/`, `p1_{split}.parquet`, `ce_*` files. See `CONTRACT.md`.

The test run ends with an inline validator (1-to-1, S1 coverage, cross-country leaks): PASS on the final submission.

## Results (same 100k val S1 for every version)

| Version | Val macro-F0.5 |
|---|---|
| empty prediction | 0.056 |
| prior attempt | 0.29 |
| v1 | 0.9612 |
| v2 | 0.9743 |
| v2 + cross-encoder base | 0.9815 |
| v3 | 0.9836 |
| v4 stage 1 | 0.9784 |
| **v4 final** | **0.9862** |

v4 validation is stricter: val S1 compete with all train S1, as in test.

Test: 1,732,544 S1, about 228M candidate pairs. Matched S1 / pairs: France 245,064 / 848,775; US 624,928 / 2,239,780; India 760,815 / 2,660,032.

## HPC Notes

- Shared GPUs: PBS does not isolate GPUs. `pipeline.pbs` pins the one with the most free memory via `CUDA_VISIBLE_DEVICES` (nvidia-smi sort on `memory.free`).
- CRLF line endings break PBS scripts (`\r` in shebang and commands). Keep `.pbs` files LF; `dos2unix` after copying from Windows.
- faiss-gpu wheels lack H100 kernels. LaBSE kNN is exact top-40 using torch fp16 matmul on the GPU (`block.emb_knn`), no faiss needed.
- Runtime (64 CPU + 1 H100): train blocking for 1.8M new S1 about 3.6 h CPU; cross-encoder fit about 1.6 h; stage-1 halves about 3 h each; full test pass about 6 h. Peak design is 2 concurrent jobs.

## References

- LaBSE (Apache-2.0), xlm-roberta-large (MIT, ~560M params), LightGBM (MIT); all under 8B params
- indic-transliteration for Indic scripts (ITRANS)
- `CONTRACT.md` for schemas, `Documentation_template.md` for methodology

## License

MIT — see [LICENSE](LICENSE). The dataset is distributed separately under Apache-2.0.
