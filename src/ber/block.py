"""Blocking / candidate generation for business entity resolution.

Per CONTRACT.md: prep frame has idx, src (1=query,2/3=pool), country, name/addr
(romanized ascii), name_orig/addr_orig (script preserved), name_key/addr_key
(phonetic skeletons), postcode, nums, etc. Blocking is always within-country.
No classes, functions only, chunked + memory aware for ~2M queries / ~10M pool.
"""

import os
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


# ---------------------------------------------------------------------------
# 1. TF-IDF char n-gram blocking, per country
# ---------------------------------------------------------------------------
def tfidf_knn(prep, text_cols, k, q_mask=None, n_jobs=64, name="tfidf_sim",
              analyzer="char_wb", ngram_range=(3, 4), max_df=None):
    """Char n-gram TF-IDF cosine top-k, computed per country.

    text_cols columns are joined with a space to form the doc text.
    analyzer/ngram_range/max_df are configurable (max_df default env BER_MAXDF, 0.02).
    Returns pl.DataFrame[s1_idx, c_idx, <name>].
    """
    if max_df is None:
        max_df = float(os.environ.get("BER_MAXDF", 0.02))
    text = pl.concat_str([pl.col(c).fill_null("") for c in text_cols], separator=" ")
    work = prep.select(
        pl.col("idx"), pl.col("src"), pl.col("country"), text.alias("_text")
    )
    if q_mask is not None:
        # q_mask is boolean over prep rows; only src==1 rows where mask True are queried
        keep_q = pl.Series("_qkeep", q_mask)
        work = work.with_columns(keep_q)
    else:
        work = work.with_columns(pl.lit(True).alias("_qkeep"))

    out_frames = []
    for country in work.select("country").unique().to_series().to_list():
        cdf = work.filter(pl.col("country") == country)
        pool = cdf.filter(pl.col("src") != 1)
        query = cdf.filter((pl.col("src") == 1) & pl.col("_qkeep"))
        if pool.height == 0 or query.height == 0:
            continue

        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=2,
            max_df=max_df,  # ponytail: frequent n-grams dominate matmul cost, carry ~no weight
            sublinear_tf=True,
            dtype=np.float32,
        )
        # fit on pool+query text of this country
        vec.fit(pl.concat([pool.select("_text"), query.select("_text")]).to_series().to_list())
        pool_mat = vec.transform(pool.select("_text").to_series().to_list()).tocsr()
        pool_idx = pool.select("idx").to_series().to_numpy()
        query_idx = query.select("idx").to_series().to_numpy()

        pool_t = pool_mat.T.tocsr()  # B must be (features x pool)
        chunk = 200_000
        rows_s1, rows_c, rows_sim = [], [], []
        for start in range(0, query.height, chunk):
            end = min(start + chunk, query.height)
            q_mat = vec.transform(
                query.select("_text").to_series().to_list()[start:end]
            ).tocsr()
            res = sp_matmul_topn(
                q_mat, pool_t, top_n=k, threshold=0.05, n_threads=n_jobs, sort=True
            )
            res = res.tocoo()
            rows_s1.append(query_idx[start:end][res.row])
            rows_c.append(pool_idx[res.col])
            rows_sim.append(res.data.astype(np.float32))

        if rows_s1:
            out_frames.append(
                pl.DataFrame(
                    {
                        "s1_idx": np.concatenate(rows_s1).astype(np.int32),
                        "c_idx": np.concatenate(rows_c).astype(np.int32),
                        name: np.concatenate(rows_sim),
                    }
                )
            )

    if not out_frames:
        return pl.DataFrame(
            schema={"s1_idx": pl.Int32, "c_idx": pl.Int32, name: pl.Float32}
        )
    return pl.concat(out_frames)


# ---------------------------------------------------------------------------
# 2. Embedding encode + FAISS kNN blocking
# ---------------------------------------------------------------------------
def emb_encode(texts, model_name, batch=2048, prefix=""):
    """Encode texts with sentence_transformers on GPU, fp16, L2-normalized.

    Sorted by length before encoding (for speed) then unsorted back.
    Returns np.float16 [n, d].
    """
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda")
    model = model.half()

    n = len(texts)
    order = sorted(range(n), key=lambda i: len(texts[i]))
    sorted_texts = [(prefix + texts[i]) if prefix else texts[i] for i in order]

    with torch.no_grad():
        emb_sorted = model.encode(
            sorted_texts,
            batch_size=batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    d = emb_sorted.shape[1]
    emb = np.empty((n, d), dtype=np.float16)
    unorder = np.empty(n, dtype=np.int64)
    unorder[order] = np.arange(n)
    emb[:] = emb_sorted[unorder].astype(np.float16)
    return emb


def emb_knn(emb, prep, k, q_mask=None, chunk=256):
    """Exact GPU top-k inner-product search per country (torch fp16 matmul; faiss-gpu wheels
    lack H100 kernels). Returns pl.DataFrame[s1_idx, c_idx, emb_sim]."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    work = prep.select("idx", "src", "country").with_columns(
        pl.Series("_qkeep", q_mask) if q_mask is not None else pl.lit(True).alias("_qkeep"))
    out = []
    for country in work["country"].unique().to_list():
        cdf = work.filter(pl.col("country") == country)
        pool_idx = cdf.filter(pl.col("src") != 1)["idx"].to_numpy()
        query_idx = cdf.filter((pl.col("src") == 1) & pl.col("_qkeep"))["idx"].to_numpy()
        if len(pool_idx) == 0 or len(query_idx) == 0:
            continue
        P = torch.from_numpy(np.asarray(emb[pool_idx])).to(dev, dt)
        kk = min(k, len(pool_idx))
        s1s, cs, sims = [], [], []
        with torch.no_grad():
            for st in range(0, len(query_idx), chunk):
                q = torch.from_numpy(np.asarray(emb[query_idx[st:st + chunk]])).to(dev, dt)
                v, i = torch.topk(q @ P.T, kk, dim=1)
                s1s.append(np.repeat(query_idx[st:st + chunk], kk))
                cs.append(pool_idx[i.cpu().numpy().ravel()])
                sims.append(v.float().cpu().numpy().ravel())
        del P
        out.append(pl.DataFrame({"s1_idx": np.concatenate(s1s).astype(np.int32),
                                 "c_idx": np.concatenate(cs).astype(np.int32),
                                 "emb_sim": np.concatenate(sims).astype(np.float32)}))
    if not out:
        return pl.DataFrame(schema={"s1_idx": pl.Int32, "c_idx": pl.Int32, "emb_sim": pl.Float32})
    return pl.concat(out)


# ---------------------------------------------------------------------------
# 3. Key-based blocking (exact match join on cheap keys)
# ---------------------------------------------------------------------------
def key_block(prep, q_mask=None, max_group=50):
    """Join queries to pool on (country, postcode) and (country, name_key
    first-token), (country, sorted name_key tokens), (country, postcode,
    first 2 sorted nums) and (country, name_key first-token[:3], first num). Skips empty keys and
    groups larger than max_group (too generic to be useful).
    Returns pl.DataFrame[s1_idx, c_idx, key_hit].
    """
    base = prep.select(
        pl.col("idx"),
        pl.col("src"),
        pl.col("country"),
        pl.col("postcode").fill_null(""),
        pl.col("name_key").fill_null(""),
        pl.col("nums").fill_null(""),
    )
    if q_mask is not None:
        base = base.with_columns(pl.Series("_qkeep", q_mask))
    else:
        base = base.with_columns(pl.lit(True).alias("_qkeep"))

    base = base.with_columns(
        pl.col("name_key").str.split(" ").alias("_toks")
    ).with_columns(
        pl.col("_toks").list.first().fill_null("").alias("_first_tok"),
        pl.col("_toks").list.sort().list.join(" ").alias("_sorted_toks"),
        pl.col("nums").fill_null("").str.split(" ").alias("_nl"),
    ).with_columns(
        pl.col("_nl").list.sort().list.head(2).list.join(" ").alias("_nums2"),
        pl.col("_nl").list.first().fill_null("").alias("_num1"),
        pl.col("_toks").list.first().fill_null("").str.slice(0, 3).alias("_first3"),
    ).with_columns(
        pl.when((pl.col("postcode") != "") & (pl.col("_nums2") != ""))
        .then(pl.col("postcode") + "|" + pl.col("_nums2")).otherwise(pl.lit("")).alias("_pc_nums"),
        pl.when((pl.col("_first3") != "") & (pl.col("_num1") != ""))
        .then(pl.col("_first3") + "|" + pl.col("_num1")).otherwise(pl.lit("")).alias("_nk3_num"),
    )

    query = base.filter((pl.col("src") == 1) & pl.col("_qkeep"))
    pool = base.filter(pl.col("src") != 1)

    def join_on(key_col):
        q = query.filter(pl.col(key_col) != "").select("idx", "country", key_col)
        p = pool.filter(pl.col(key_col) != "").select("idx", "country", key_col)
        # drop groups on the pool side that are too big to be a useful block
        grp_sizes = p.group_by(["country", key_col]).len()
        p = p.join(
            grp_sizes.filter(pl.col("len") <= max_group).drop("len"),
            on=["country", key_col],
            how="inner",
        )
        j = q.join(p, on=["country", key_col], suffix="_pool")
        return j.select(
            pl.col("idx").alias("s1_idx"), pl.col("idx_pool").alias("c_idx")
        )

    parts = [
        join_on("postcode"),
        join_on("_first_tok"),
        join_on("_sorted_toks"),
        join_on("_pc_nums"),
        join_on("_nk3_num"),
    ]
    hits = pl.concat(parts).unique()
    return hits.with_columns(pl.lit(1, dtype=pl.Int8).alias("key_hit"))


# ---------------------------------------------------------------------------
# 4. Union candidate pairs from multiple blocking methods
# ---------------------------------------------------------------------------
def union_pairs(*dfs):
    """Outer-join candidate frames on (s1_idx, c_idx). Missing sim cols become
    NaN, missing key_hit becomes 0."""
    dfs = [d for d in dfs if d.height > 0]
    if not dfs:
        return pl.DataFrame(schema={"s1_idx": pl.Int32, "c_idx": pl.Int32})

    out = dfs[0]
    for d in dfs[1:]:
        out = out.join(d, on=["s1_idx", "c_idx"], how="full", coalesce=True)

    sim_cols = [c for c in out.columns if c not in ("s1_idx", "c_idx", "key_hit")]
    exprs = [pl.col(c).fill_null(np.nan) for c in sim_cols]
    if "key_hit" in out.columns:
        exprs.append(pl.col("key_hit").fill_null(0).cast(pl.Int8))
    out = out.with_columns(exprs)
    return out


# ---------------------------------------------------------------------------
# 5. Recall diagnostics against ground truth
# ---------------------------------------------------------------------------
def recall_report(pairs, gt, s1_subset=None):
    """Report pair recall, fraction of S1 fully covered, candidate counts,
    and recall@k truncation curves per sim column.
    """
    if s1_subset is not None:
        subset = pl.Series("s1_idx", s1_subset)
        gt_eval = gt.filter(pl.col("s1_idx").is_in(subset))
        pairs_eval = pairs.filter(pl.col("s1_idx").is_in(subset))
    else:
        gt_eval = gt
        pairs_eval = pairs

    gt_keys = gt_eval.select("s1_idx", "c_idx").unique()
    n_gt = gt_keys.height
    found = gt_keys.join(pairs_eval.select("s1_idx", "c_idx").unique(), on=["s1_idx", "c_idx"], how="inner")
    pair_recall = found.height / n_gt if n_gt else float("nan")

    gt_per_s1 = gt_keys.group_by("s1_idx").agg(pl.len().alias("n_true"))
    found_per_s1 = found.group_by("s1_idx").agg(pl.len().alias("n_found"))
    cov = gt_per_s1.join(found_per_s1, on="s1_idx", how="left").with_columns(
        pl.col("n_found").fill_null(0)
    )
    frac_full_cov = (cov["n_found"] == cov["n_true"]).mean() if cov.height else float("nan")

    cand_counts = pairs_eval.group_by("s1_idx").agg(pl.len().alias("n_cand"))["n_cand"]
    mean_cand = float(cand_counts.mean()) if cand_counts.len() else 0.0
    median_cand = float(cand_counts.median()) if cand_counts.len() else 0.0

    sim_cols = [
        c for c in pairs_eval.columns if c not in ("s1_idx", "c_idx", "key_hit")
    ]
    recall_at_k = {}
    for col in sim_cols:
        curve = {}
        ranked = pairs_eval.filter(pl.col(col).is_not_nan()).with_columns(
            pl.col(col).rank(method="ordinal", descending=True).over("s1_idx").alias("_rank")
        )
        for k in (5, 10, 20, 30, 50):
            topk = ranked.filter(pl.col("_rank") <= k).select("s1_idx", "c_idx").unique()
            f = gt_keys.join(topk, on=["s1_idx", "c_idx"], how="inner")
            curve[k] = f.height / n_gt if n_gt else float("nan")
        recall_at_k[col] = curve

    return {
        "pair_recall": pair_recall,
        "frac_s1_fully_covered": frac_full_cov,
        "mean_cand_per_s1": mean_cand,
        "median_cand_per_s1": median_cand,
        "recall_at_k": recall_at_k,
    }


# ---------------------------------------------------------------------------
# 6. Cap candidate pairs per query
# ---------------------------------------------------------------------------
def cap_pairs(pairs, k_total, per_method=25):
    """Rank-fusion cap. Per s1_idx, rank each *_sim column (desc, null/NaN -> inf).
    Keep key_hit==1 rows or rows with min_rank <= per_method; if more than k_total
    remain per s1, keep the best k_total by (min_rank asc, sum 1/rank desc).
    key_hit rows are always kept."""
    sims = [c for c in pairs.columns if c.endswith("_sim")]
    if not sims:
        return pairs.with_columns(pl.int_range(pl.len()).over("s1_idx").alias("_r")).filter(
            (pl.col("_r") < k_total) | ((pl.col("key_hit") == 1) if "key_hit" in pairs.columns else pl.lit(False))
        ).drop("_r")
    inf = float("inf")
    ranks = [
        pl.col(c).fill_nan(None).rank(method="min", descending=True).over("s1_idx")
        .cast(pl.Float64).fill_null(inf).alias(f"_rk_{c}")
        for c in sims
    ]
    rk = [f"_rk_{c}" for c in sims]
    df = pairs.with_columns(ranks).with_columns(
        pl.min_horizontal(rk).alias("_minr"),
        pl.sum_horizontal([1.0 / pl.col(r) for r in rk]).alias("_rs"),
    )
    hit = (pl.col("key_hit") == 1) if "key_hit" in pairs.columns else pl.lit(False)
    df = df.filter(hit | (pl.col("_minr") <= per_method)).sort(
        ["s1_idx", "_minr", "_rs"], descending=[False, False, True]
    ).with_columns(pl.int_range(pl.len()).over("s1_idx").alias("_pos"))
    return df.filter(hit | (pl.col("_pos") < k_total)).drop(rk + ["_minr", "_rs", "_pos"])


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import string

    os.environ.setdefault("BER_MAXDF", "0.5")  # tiny synthetic data
    random.seed(0)
    np.random.seed(0)

    def rand_name():
        return "".join(random.choices(string.ascii_lowercase, k=8))

    n_pool = 150
    n_query = 50
    countries = ["US", "IN"]
    rows = []
    idx = 0
    entity_names = {}
    for i in range(n_pool):
        c = random.choice(countries)
        nm = rand_name()
        pc = str(random.randint(10000, 99999))
        rows.append(
            dict(
                idx=idx, entity_id=f"e{i}", src=random.choice([2, 3]), country=c,
                name_orig=nm, addr_orig=f"{nm} street {pc}",
                name=nm, addr=f"{nm} street {pc}", core=nm, legal="",
                name_key=nm[:4], addr_key=nm[:4], postcode=pc, nums=pc,
                script="latin", landmark="",
            )
        )
        entity_names[i] = (nm, c, pc)
        idx += 1

    gt_rows = []
    for i in range(n_query):
        base_i = random.randrange(n_pool)
        nm, c, pc = entity_names[base_i]
        rows.append(
            dict(
                idx=idx, entity_id=f"q{i}", src=1, country=c,
                name_orig=nm, addr_orig=f"{nm} street {pc}",
                name=nm, addr=f"{nm} street {pc}", core=nm, legal="",
                name_key=nm[:4], addr_key=nm[:4], postcode=pc, nums=pc,
                script="latin", landmark="",
            )
        )
        # find the pool row idx matching base_i (pool rows are 0..n_pool-1 == their idx)
        gt_rows.append(dict(s1_idx=idx, c_idx=base_i))
        idx += 1

    prep = pl.DataFrame(rows)
    gt = pl.DataFrame(gt_rows).with_columns(pl.col("s1_idx", "c_idx").cast(pl.Int32))
    prep = prep.with_columns(pl.col("idx").cast(pl.Int32))

    t = tfidf_knn(prep, ["name", "addr"], k=10, name="tfidf_sim")
    tn = tfidf_knn(prep, ["name"], k=10, name="tfidf_name_sim")
    kb = key_block(prep, max_group=50)
    print("tfidf pairs:", t.height, "name pairs:", tn.height, "key pairs:", kb.height)

    pairs = union_pairs(t, tn, kb)
    print("union pairs:", pairs.height, pairs.columns)

    capped = cap_pairs(pairs, k_total=5)
    print("capped pairs:", capped.height)
    assert capped.columns == pairs.columns
    # key_hit rows survive; non-key rows capped per s1
    nk = capped.filter(pl.col("key_hit") == 0).group_by("s1_idx").len()
    assert nk.height == 0 or nk["len"].max() <= 5

    # new tfidf kwargs
    tk = tfidf_knn(prep, ["name_key"], k=10, name="tfidf_key_sim", ngram_range=(2, 3))
    ta = tfidf_knn(prep, ["addr"], k=10, name="tfidf_addr_sim", analyzer="char_wb", ngram_range=(3, 4), max_df=0.5)
    # fake emb_sim: high-baseline cosine-ish scale
    fe = t.select("s1_idx", "c_idx").with_columns(
        (pl.Series(np.random.uniform(0.6, 0.95, t.height)).cast(pl.Float32)).alias("emb_sim"))
    p5 = union_pairs(t, tn, tk, ta, fe, kb)
    sc = [c for c in p5.columns if c.endswith("_sim")]
    assert sc == ["tfidf_sim", "tfidf_name_sim", "tfidf_key_sim", "tfidf_addr_sim", "emb_sim"], sc
    assert p5.height >= t.height
    c5 = cap_pairs(p5, k_total=6, per_method=3)
    assert c5.height <= p5.height and c5.columns == p5.columns
    assert c5.join(kb, on=["s1_idx", "c_idx"]).height == kb.height  # key hits kept
    print("5-col union", p5.height, "capped", c5.height)

    rr = recall_report(pairs, gt)
    print("recall_report:", rr)
    assert rr["pair_recall"] > 0.8, "expected high recall on synthetic exact-ish data"

    print("block.py smoke test OK")


# ---------------------------------------------------------------------------
# Learned cap: stage-0 ranker over the union (sims + per-method ranks only)
# ---------------------------------------------------------------------------
def _stage0_frame(pairs):
    sims = sorted(c for c in pairs.columns if c.endswith("_sim"))
    exprs = []
    for c in sims:
        v = pl.col(c).fill_nan(None)
        exprs += [v.fill_null(-1).alias(f"{c}_v"),
                  v.rank("min", descending=True).over("s1_idx").fill_null(999).alias(f"{c}_r1"),
                  v.rank("min", descending=True).over("c_idx").fill_null(999).alias(f"{c}_rc")]
    exprs += [pl.sum_horizontal([pl.col(c).fill_nan(None).is_not_null() for c in sims]).alias("n_methods"),
              pl.col("key_hit").cast(pl.Int8), pl.len().over("s1_idx").alias("n_union_s1")]
    return pairs.select("s1_idx", "c_idx", *exprs)


def learned_cap(pairs, k_total, model_path, gt=None, train_s1=None, n_train_s1=100_000, per_method=10):
    """Keep top k_total per S1 by a small LightGBM trained on union pairs of train_s1 (labels from gt).
    Hybrid: also keep every method's top-per_method and key hits (hard pairs look like non-matches on sims,
    so a pure learned cap drops them; val recall 0.9802 @129 pairs/S1 vs 0.9712 @150 learned-only).
    With gt=None the saved model at model_path is applied (test)."""
    import lightgbm as lgb
    X = _stage0_frame(pairs)
    cols = [c for c in X.columns if c not in ("s1_idx", "c_idx")]
    if gt is not None:
        rng = np.random.default_rng(0)
        pick = rng.choice(np.asarray(train_s1), min(n_train_s1, len(train_s1)), replace=False)
        tr = X.filter(pl.col("s1_idx").is_in(pl.Series(pick).implode()))
        y = tr.join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1_idx", "c_idx"], how="left")["y"].fill_null(0)
        bst = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
                         "verbose": -1, "num_threads": int(os.environ.get("OMP_NUM_THREADS", 64))},
                        lgb.Dataset(tr.select(cols).to_numpy(), y.to_numpy()), num_boost_round=300)
        bst.save_model(str(model_path))
    else:
        bst = lgb.Booster(model_file=str(model_path))
    p0 = bst.predict(X.select(cols).to_numpy(), num_threads=int(os.environ.get("OMP_NUM_THREADS", 64)))
    mr = pl.min_horizontal([pl.col(c) for c in X.columns if c.endswith("_r1")])
    keep = (X.select("s1_idx", "key_hit", mr.alias("_mr")).with_columns(pl.Series("p0", p0, pl.Float32))
             .select((pl.col("p0").rank("ordinal", descending=True).over("s1_idx") <= k_total)
                     | (pl.col("_mr") <= per_method) | (pl.col("key_hit") == 1))).to_series()
    return pairs.filter(keep.to_numpy()).with_columns(pl.Series("p0", p0[keep.to_numpy()], pl.Float32))
