# Data contract (shared by all src/ber modules)

Paths: repo root = ber/. Data: `$BER_DATA` (default ~/amazonml/amazon-ml/dataset/{train,test}/*.tsv, tab-sep, cols entity_id,business_name,business_address,country; GT: source1_entity_id, matched_entity_ids comma-sep, may be empty/NaN). Artifacts: `$BER_WORK` (default ber/working/). Submission: `$BER_OUT`.

## {split}_prep.parquet  (split in {train,test}; rows = S1 ++ S2 ++ S3 concatenated)
idx int32 (0..n-1, row position) | entity_id str | src int8 (1/2/3) | country str (open set, never hardcode)
name_orig, addr_orig : NFKC, lowercased, ws-collapsed, script preserved (embedding / cross-encoder input)
name, addr           : romanized ASCII view (ITRANS + anusvara + schwa deletion), leet de-noised, web junk stripped, leading zeros normalized, abbreviations expanded, punctuation stripped
core                 : `name` minus legal suffixes / generic filler
legal                : canonical legal suffix ("" if none) e.g. "pvt ltd","llc","sarl"
name_key, addr_key   : consonant-skeleton phonetic keys (cross-script / transliteration bridge); sorted-token variants used for key joins
name_key2            : coarse voicing-blind sound-class key of core (p/b/f/v/w=P, t/d=T, k/g/c=K, s/z/j/sh=S, m/n=N, h+vowels dropped); bridges Tamil script, which has no t/d, k/g, p/b distinction
postcode             : str ("" none) ; nums: space-joined other numbers in address
script               : dominant script of raw name: latin|devanagari|bengali|tamil|telugu|gujarati|kannada|malayalam|gurmukhi|oriya|arabic|other
landmark             : landmark phrase pulled from address ("" none)

## train_gt.parquet : s1_idx int32, c_idx int32  (row idx into train_prep)
## train_folds.parquet : s1_idx int32, fold str in {va, ce, tr}
va = first 100k (N_VA) of a fixed S1 permutation (same val in every version; tuning/early stopping), ce = next 500k (N_CE; cross-encoder fit only), tr = rest (1,606,821; stage 1 and 2 fit). All train S1 are blocked and featurized regardless of fold.
## {split}_emb_{tag}.npy : float16 [n, d], L2-normalized, row = idx
## {split}_blk_{method}.parquet : s1_idx, c_idx int32 + the method's score; method in tfidf|name|skel|coarse|addr|emb|key (append mode: already-blocked S1 are skipped). Score cols: tfidf_sim, tfidf_name_sim, tfidf_key_sim, tfidf_coarse_sim, tfidf_addr_sim, emb_sim (float32, NaN if method missed); key_hit int8
## {split}_feats_{i}.parquet : i in 0..N_CHUNKS-1, chunked by hash(s1_idx) % N_CHUNKS. Capped union pairs (stage-0 ranker top-K_TOTAL ∪ each method's top-K_PER ∪ key hits) + sims, global competition ranks, pair features (+ label int8 for train). Read all via glob `{split}_feats_*.parquet`.
## p1_{split}.parquet : s1_idx, c_idx, p float32 (stage-1 probability for EVERY pair; train tr S1 get the out-of-fold half model, other folds the average of both)
## ce_fit_pairs.parquet : s1_idx, c_idx, label (cross-encoder fit pairs from fold ce: positives + top-10 stage-0 negatives)
## ce_model/ : fine-tuned cross-encoder (FacebookAI/xlm-roberta-large)
## ce_{split}_pairs.parquet : s1_idx, c_idx (pairs with stage-1 p >= CE_BAND) -> ce_{split}_scores.parquet : s1_idx, c_idx, ce float32 (pairs below the band have no score, null after left join)
## lgb0.txt (stage-0 cap ranker), lgb1_0.txt / lgb1_1.txt (stage-1, OOF halves by hash(s1_idx)%2), lgb2.txt (stage 2, pairs with p1 >= S2_MIN; adds ce, anchor features, p competition ranks)
## train_result.json : t, alpha, val_f05, by_country, by_script, features (stage 1), features2 (stage 2), importance, stage1 results; tune_table.csv; val_pred.parquet; blocking_report.json
## $BER_OUT/matching_results.tsv, candidate_pairs.tsv : s1 entity id -> comma-separated matched / candidate ids
Pools: queries = src==1 rows; pool = src in (2,3); always same-country only.
