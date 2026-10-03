# Business Entity Resolution: Methodology & Design (final v4)

Amazon ML Challenge 2026: macro-F0.5 matching of business entities across three sources, multilingual names, 1-to-1 constraint on pool entities, unseen France in test.

**Final result: validation macro-F0.5 0.9862** (100k held-out S1; India 0.9823, US 0.9887; t=0.425, alpha=0.7). Test: 1,732,544 S1, about 228M scored candidate pairs, validator PASS.

## 1. Methodology Overview

Task: for each source-1 (S1) entity, find matching entities in source 2 / source 3 (S2/S3) of the same country. Metric: macro-F0.5 per S1 (precision weighted over recall). Rules: no cross-country matches, each pool entity matched to at most one S1, France appears only at test.

Pipeline:
1. Text normalization into two views (original script and romanized).
2. Per-country candidate generation: TF-IDF char kNN (several keys), LaBSE exact kNN, key joins; hybrid learned cap.
3. Stage-1 LightGBM on string/phonetic/structural features.
4. Cross-encoder (xlm-roberta-large) fine-tuned on S1/candidate pairs.
5. Stage-2 LightGBM adding cross-encoder score and competition features.
6. 1-to-1 decision with tuned thresholds.

No external data or APIs; everything is learned from the challenge data plus public pretrained models.

## 2. Data Insights

- Names appear in many scripts (Latin, Devanagari, Tamil, Telugu, Bengali, ...) and S2 often holds a transliterated variant of the S1 name.
- Tamil script has no t/d, k/g, p/b distinction, so romanized forms of the same name differ in voicing.
- Noise: leetspeak in names (0 for o, etc.), web junk (urls, domains), inconsistent leading zeros in numbers, abbreviations, legal-suffix variation.
- All true matches are same-country; the blocking filter is per country.
- Pool-side competition matters: in test each pool record sees about 22 competing S1 on average; blocking only a fold of S1 gave about 5. v4 blocks every train S1 so training and test competition agree.

## 3. Preprocessing (`text.py`)

Two views per entity:
- Original script (`name_orig`, `addr_orig`): NFKC, lowercase, whitespace-collapsed, script kept. Input to LaBSE embeddings and the cross-encoder.
- Romanized (`name`, `addr`): used for string similarities and TF-IDF.

Romanization and cleanup:
- Indic digits to ASCII; Indic transliteration via ITRANS (indic-transliteration), then unidecode.
- Anusvara resolved to the homorganic nasal; schwa deletion; candra vowels mapped to plain vowels.
- Leet de-noising in names (0 -> o etc.), web junk stripping, leading-zero normalization in numbers.
- Abbreviation expansion, legal-suffix extraction (`legal`, `core` = name without suffix/filler).

Keys:
- `name_key`: consonant skeleton (cross-script bridge).
- `name_key2`: coarse voicing-blind sound-class key (p/b/f/v/w, t/d, k/g/c, s/z/j/sh, m/n collapsed; h and vowels dropped), built for Tamil-script names.
- Sorted-token variants of the keys for order-insensitive joins.
- Also `postcode`, `nums`, `landmark`, `script`.

## 4. Candidate Generation / Blocking (`block.py`)

Per country, same-country pairs only. Methods:

| Method | Text | Depth |
|---|---|---|
| TF-IDF char kNN | name + addr | k=40 |
| TF-IDF char kNN | core name | k=60 |
| TF-IDF char kNN | skeleton key (`name_key`) | k=60 |
| TF-IDF char kNN | coarse key (`name_key2`) | k=60 |
| TF-IDF char kNN | address | k=20 |
| LaBSE embedding kNN | `name_orig` + `addr_orig` | exact top-40 |
| Key joins | postcode / name keys / sorted keys | exact, skip oversized groups |

LaBSE kNN is exact: embeddings are L2-normalized fp16 and scored with a torch fp16 matmul on the GPU. The faiss-gpu wheel lacks H100 kernels, so faiss is not used.

Union recall on validation S1 is 0.9826.

### Hybrid cap

The union is capped per S1 to keep the feature stage tractable: a stage-0 LightGBM ranker (similarities and ranks) keeps its top-60, united with each method's top-10 and all key hits. Result on val S1: pair recall 0.9803, 93.8% of S1 fully covered, mean 126.7 candidates per S1 (median 127). A learned-only cap was worse (recall 0.9712 at 150): hard true pairs look like non-matches on similarity alone, so per-method top-10 and key hits protect them.

Recall at k (val S1, selected):

| Method | k=10 | k=20 | k=50 |
|---|---|---|---|
| tfidf (name+addr) | 0.9117 | 0.9356 | 0.9440 |
| tfidf name (core) | 0.5095 | 0.5828 | 0.6465 |
| tfidf skeleton key | 0.4573 | 0.5080 | 0.5469 |
| tfidf coarse key | 0.4151 | 0.4603 | 0.4931 |
| tfidf address | 0.8015 | 0.8375 | 0.8375 |
| LaBSE embedding | 0.8491 | 0.8786 | 0.8800 |
| stage-0 p0 | 0.9390 | 0.9517 | 0.9618 |

Single methods are individually weak (the key methods especially); their value is catching pairs the others miss, which the union does.

All 2.2M train S1 are blocked (append mode reuses earlier blocks), so pool-side competition in training matches test.

## 5. Model Architecture & Feature Engineering

### Folds
- `va`: first 100k of a fixed S1 permutation (identical across all versions; tuning and early stopping).
- `ce`: next 500k, used only to fit the cross-encoder.
- `tr`: remaining 1,606,821, used for LightGBM.

### Features (`features.py`)
- String similarities (RapidFuzz ratio/token-set/token-sort/partial/Jaro-Winkler) on name, core, address and name+address.
- Phonetic key similarities on `name_key`, `addr_key` (and coarse key).
- Token/structural: Jaccard, per-country IDF-weighted Jaccard, legal-suffix agreement, postcode agreement, number-set similarity, landmark flags.
- Blocking similarities (tfidf, name, skeleton, coarse, address, embedding) with global competition ranks over all S1.
- Counts of same-name / same-address pairs per S1.

### Stage 1 LightGBM
Two out-of-fold halves split by S1 hash; each fit on 350k tr S1 (about 44.4M rows, 1.19M positives), early stopped on val. Out-of-fold prediction for every train and test pair (tr S1 get the model that did not see them). Stage 1 alone: val 0.9784.

### Cross-encoder (`ce.py`)
- Base: FacebookAI/xlm-roberta-large (MIT, ~560M params). Input is the pair "name_orig (romanized) | addr_orig" for the S1 and the candidate.
- Fit on the `ce` fold: 5,067,775 pairs (33.5% positive), all positives plus hard negatives (top-10 by stage-0 score).
- 1 epoch, bf16, batch 64, lr 1e-5, about 900 pairs/s.
- Scored on pairs with stage-1 p >= 0.005.

### Stage 2 LightGBM
Fit on pairs with p1 >= 0.001 (10.5M rows over all S1; 7.63M in tr). Adds the cross-encoder score, cluster-consistency anchor features, and p competition ranks/gaps per S1 and per candidate. Top features by gain: `ce`, `p1`, `p_rank_c`, `p_gap_s1max`, `n_cands_c`, then `p_gap_cmax`, `nums_jaccard`, `s2_addr_tset`, `anchor_p_max`, `s2_name_tset`.

### Decision layer (`model.py`)
1. 1-to-1 argmax per S2/S3 record over all S1 (the S1 with highest p keeps it).
2. Keep a pair iff p >= t and p >= alpha x (max p of that S1).
3. t and alpha tuned on val macro-F0.5: t=0.425, alpha=0.7.

## 6. Other Information

### Unseen France
The model is country-agnostic: country is used only for the same-country blocking filter and never as a feature. LaBSE and xlm-roberta-large are multilingual; abbreviation/legal-suffix maps include French forms (SARL, EURL, ...); IDF is computed per country from the data itself. France test output: 245,064 matched S1 / 848,775 pairs.

### Compliance
- No external data, databases or APIs.
- Models: LaBSE (Apache-2.0), xlm-roberta-large (MIT), LightGBM (MIT); all at most 8B parameters.
- Inline validator (`model.validate`) checks 1-to-1, S1 coverage and cross-country leaks: PASS.

### Test output
1,732,544 S1; about 228M candidate pairs. Matched S1 / pairs: France 245,064 / 848,775; US 624,928 / 2,239,780; India 760,815 / 2,660,032. Files: `matching_results.tsv` (97 MB), `candidate_pairs.tsv` (3.0 GB).

### Runtime (64 CPU + 1 H100, PBS)
Train blocking for 1.8M new S1 about 3.6 h CPU; train features 279.5M pairs; cross-encoder fit about 1.6 h; stage-1 halves about 3 h each; full test pass about 6 h. Designed for 2 concurrent jobs, chained with `qsub -W depend=afterok`.

### Iteration history (same 100k val S1)

| Version | Change | Val macro-F0.5 |
|---|---|---|
| empty | no matches | 0.056 |
| prior attempt | exact/prefix blocking + LightGBM | 0.29 |
| v1 | TF-IDF + LaBSE + key blocking (cap 60), LightGBM, 1-to-1 assignment | 0.9612 |
| v2 | transliteration fixes (anusvara), leet/web/number de-noising, skeleton + address kNN, rank-fusion cap, rarity features, stage-2 anchor model | 0.9743 |
| v2 + CE base | xlm-roberta-base cross-encoder score in stage 2 | 0.9815 |
| v3 | coarse voicing-blind key + kNN, wider name kNN, hybrid learned cap, xlm-roberta-large CE | 0.9836 |
| v4 stage 1 | all 2.2M train S1 blocked (test-like competition), 2.6x more training S1, chunked features | 0.9784 |
| **v4 final** | **stage 2 + xlm-roberta-large CE fit on a disjoint 500k-S1 fold** | **0.9862** |

v4 validation is stricter than earlier versions: val S1 compete with every train S1, as in test. Blocking ceiling on val is about 0.9935.

## 7. Limitations & Next Steps

- Residual gap to blocking ceiling (about 0.9935 vs 0.9862): hard pairs (heavy transliteration drift, sparse names) rank low on both similarity and CE.
- Schwa deletion and homorganic nasal rules are heuristics; language-specific grapheme-to-phoneme models could improve romanization.
- Cap recall 0.9803: pairs outside every method's top-k and the key joins are lost; deeper per-method k raises cost.
- Cross-encoder trained 1 epoch on one fold; more epochs or a second pass on stage-2 hard negatives may help.
- Single global t and alpha; per-country or per-script thresholds are untested.
- Compute: test pass is about 6 h and candidate_pairs is 3.0 GB; a smaller cap or CE band would trade accuracy for speed.
