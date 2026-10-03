"""Pairwise feature engineering for business entity resolution.

Consumes candidate pairs (s1_idx, c_idx, ...) from block.py and the prep
frame (see CONTRACT.md) and produces float32 feature columns for a
classifier. No classes; multiprocessing.Pool(fork) over chunks with prep
columns preloaded into module globals before forking (fork gives workers a
copy-on-write view of the globals, no pickling of the whole prep frame per
task).
"""

import math
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# module-level globals populated by _set_globals() before Pool(fork) spawn.
# Plain python lists indexed by row idx (contract: idx is 0..n-1 row position).
_G = {}


def _tok(s):
    return s.split(" ") if s else []


def _build_idf(prep):
    """Per-country IDF over `name` tokens, used to weight core-token jaccard.
    Returns {country: {token: idf}} plus a per-country default idf for OOV
    tokens (log(N) i.e. as if df=1).
    """
    idf_by_country = {}
    for country in prep.select("country").unique().to_series().to_list():
        names = prep.filter(pl.col("country") == country).select("name").to_series().to_list()
        n_docs = len(names)
        df = Counter()
        for nm in names:
            for t in set(_tok(nm)):
                df[t] += 1
        idf = {t: math.log(n_docs / (1 + c)) for t, c in df.items()}
        # top-5% idf threshold (rare-token cutoff); OOV tokens (default idf) also exceed it
        idf["__p95__"] = float(np.percentile(list(idf.values()), 95)) if idf else float("inf")
        idf["__default__"] = math.log(max(n_docs, 1))
        idf_by_country[country] = idf
    return idf_by_country


def _set_globals(prep, idf_by_country=None):
    """Stash prep columns as plain lists (fast, fork-shareable) in _G."""
    global _G
    _G = {
        "name": prep["name"].to_list(),
        "addr": prep["addr"].to_list(),
        "core": prep["core"].to_list(),
        "legal": prep["legal"].to_list(),
        "name_key": prep["name_key"].to_list(),
        "addr_key": prep["addr_key"].to_list(),
        "postcode": prep["postcode"].to_list(),
        "nums": prep["nums"].to_list(),
        "script": prep["script"].to_list(),
        "landmark": prep["landmark"].to_list(),
        "src": prep["src"].to_list(),
        "idf": idf_by_country or {},
        "country": prep["country"].to_list(),
    }


def _jaccard(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _idf_jaccard(a_tokens, b_tokens, idf_map):
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    default = idf_map.get("__default__", 1.0)
    inter_w = sum(idf_map.get(t, default) for t in (a & b))
    union_w = sum(idf_map.get(t, default) for t in (a | b))
    return inter_w / union_w if union_w > 0 else 1.0

def _num_set(s):
    return set(s.split(" ")) if s else set()


def _process_pair(qi, ci):
    """Compute one row of features (dict) for query row qi vs candidate row ci."""
    g = _G
    q_name, c_name = g["name"][qi], g["name"][ci]
    q_core, c_core = g["core"][qi], g["core"][ci]
    q_addr, c_addr = g["addr"][qi], g["addr"][ci]
    q_full, c_full = q_name + " " + q_addr, c_name + " " + c_addr

    feats = {}

    def add_field(prefix, qs, cs, with_partial_jw=True):
        feats[f"{prefix}_ratio"] = fuzz.ratio(qs, cs) / 100.0
        feats[f"{prefix}_tset"] = fuzz.token_set_ratio(qs, cs) / 100.0
        feats[f"{prefix}_tsort"] = fuzz.token_sort_ratio(qs, cs) / 100.0
        if with_partial_jw:
            feats[f"{prefix}_partial"] = fuzz.partial_ratio(qs, cs) / 100.0
            feats[f"{prefix}_jw"] = JaroWinkler.normalized_similarity(qs, cs)

    add_field("name", q_name, c_name)
    add_field("core", q_core, c_core)
    add_field("addr", q_addr, c_addr, with_partial_jw=False)
    add_field("full", q_full, c_full, with_partial_jw=False)

    # phonetic keys
    qk, ck = g["name_key"][qi], g["name_key"][ci]
    feats["namekey_ratio"] = fuzz.ratio(qk, ck) / 100.0
    feats["namekey_tset"] = fuzz.token_set_ratio(qk, ck) / 100.0
    qak, cak = g["addr_key"][qi], g["addr_key"][ci]
    feats["addrkey_ratio"] = fuzz.ratio(qak, cak) / 100.0
    feats["addrkey_tset"] = fuzz.token_set_ratio(qak, cak) / 100.0

    # token jaccard
    q_core_tok, c_core_tok = _tok(q_core), _tok(c_core)
    q_addr_tok, c_addr_tok = _tok(q_addr), _tok(c_addr)
    feats["core_jaccard"] = _jaccard(q_core_tok, c_core_tok)
    feats["addr_jaccard"] = _jaccard(q_addr_tok, c_addr_tok)
    country = g["country"][qi]
    idf_map = g["idf"].get(country, {"__default__": 1.0})
    feats["core_idf_jaccard"] = _idf_jaccard(q_core_tok, c_core_tok, idf_map)

    # rarity
    dflt = idf_map.get("__default__", 1.0)
    q_idf = [idf_map.get(t, dflt) for t in q_core_tok]
    c_idf = [idf_map.get(t, dflt) for t in c_core_tok]
    feats["q_core_max_idf"] = max(q_idf) if q_idf else 0.0
    feats["c_core_max_idf"] = max(c_idf) if c_idf else 0.0
    feats["q_core_mean_idf"] = sum(q_idf) / len(q_idf) if q_idf else 0.0
    feats["c_core_mean_idf"] = sum(c_idf) / len(c_idf) if c_idf else 0.0
    shared = set(q_core_tok) & set(c_core_tok)
    feats["shared_rare_token"] = max(idf_map.get(t, dflt) for t in shared) if shared else 0.0
    feats["q_name_ntok"] = len(_tok(q_name))
    feats["c_name_ntok"] = len(_tok(c_name))
    feats["q_addr_empty"] = np.int8(1 if len(q_addr) < 5 else 0)
    feats["c_addr_empty"] = np.int8(1 if len(c_addr) < 5 else 0)
    feats["c_name_is_single_token_oov"] = np.int8(
        1 if len(c_core_tok) == 1 and c_idf[0] >= idf_map.get("__p95__", float("inf")) else 0)

    # legal suffix
    q_legal, c_legal = g["legal"][qi], g["legal"][ci]
    feats["legal_both_empty"] = np.int8(1 if (not q_legal and not c_legal) else 0)
    feats["legal_same"] = np.int8(1 if (q_legal and c_legal and q_legal == c_legal) else 0)
    feats["legal_diff"] = np.int8(1 if (q_legal and c_legal and q_legal != c_legal) else 0)

    # postcode
    q_pc, c_pc = g["postcode"][qi], g["postcode"][ci]
    feats["postcode_eq"] = np.int8(1 if (q_pc and c_pc and q_pc == c_pc) else 0)
    feats["postcode_missing"] = np.int8(1 if (not q_pc or not c_pc) else 0)
    feats["postcode_diff"] = np.int8(1 if (q_pc and c_pc and q_pc != c_pc) else 0)

    # nums
    # leading zeros stripped so '04348' == '4348'
    q_nl = [t.lstrip("0") or "0" for t in g["nums"][qi].split(" ")] if g["nums"][qi] else []
    c_nl = [t.lstrip("0") or "0" for t in g["nums"][ci].split(" ")] if g["nums"][ci] else []
    feats["nums_jaccard"] = _jaccard(q_nl, c_nl)
    first_eq = bool(q_nl) and bool(c_nl) and q_nl[0] == c_nl[0]
    feats["nums_first_eq"] = np.int8(1 if first_eq else 0)
    feats["house_num_eq"] = np.int8(1 if first_eq else 0)
    feats["num_overlap_any"] = np.int8(1 if (set(q_nl) & set(c_nl)) else 0)
    feats["addr_tset_x_num"] = feats["addr_tset"] * (1.0 if first_eq else 0.0)

    # landmark
    feats["landmark_q_present"] = np.int8(1 if g["landmark"][qi] else 0)
    feats["landmark_c_present"] = np.int8(1 if g["landmark"][ci] else 0)

    # length features
    len_q_name, len_c_name = len(q_name), len(c_name)
    feats["len_ratio_name"] = (min(len_q_name, len_c_name) / max(len_q_name, len_c_name)) if max(len_q_name, len_c_name) else 1.0
    len_q_addr, len_c_addr = len(q_addr), len(c_addr)
    feats["len_ratio_addr"] = (min(len_q_addr, len_c_addr) / max(len_q_addr, len_c_addr)) if max(len_q_addr, len_c_addr) else 1.0
    feats["n_tokens_diff"] = abs(len(_tok(q_full)) - len(_tok(c_full)))

    # script
    q_script, c_script = g["script"][qi], g["script"][ci]
    feats["same_script"] = np.int8(1 if q_script == c_script else 0)
    feats["q_nonlatin"] = np.int8(0 if q_script == "latin" else 1)
    feats["c_nonlatin"] = np.int8(0 if c_script == "latin" else 1)
    feats["c_src"] = np.int8(g["src"][ci])

    return feats


def _process_chunk(args):
    s1_chunk, c_chunk = args
    rows = [_process_pair(int(qi), int(ci)) for qi, ci in zip(s1_chunk, c_chunk)]
    return rows


def _pmap(fn, chunks, n_jobs):
    """fork Pool.imap over chunks (globals set before fork); in-process fallback."""
    try:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(n_jobs) as pool:
            return list(pool.imap(fn, chunks))
    except (ValueError, ImportError, OSError):
        # fork not available (e.g. Windows dev box) -> run in-process
        return [fn(c) for c in chunks]


def build_features(pairs, prep, emb=None, n_jobs=64, chunk=500_000):
    """Compute feature columns for every (s1_idx, c_idx) pair. Returns
    pairs joined with float32 feature columns (+ emb_cos if emb given).
    Country is deliberately excluded from the output features.
    """
    idf_by_country = _build_idf(prep)
    _set_globals(prep, idf_by_country)

    s1_arr = pairs["s1_idx"].to_numpy()
    c_arr = pairs["c_idx"].to_numpy()
    n = len(s1_arr)

    chunks = [
        (s1_arr[i : i + chunk], c_arr[i : i + chunk]) for i in range(0, n, chunk)
    ]

    all_rows = [r for rows in _pmap(_process_chunk, chunks, n_jobs) for r in rows]

    feat_df = pl.DataFrame(all_rows)
    # cast everything float32 except the explicit int8 flag columns
    int8_cols = [c for c in feat_df.columns if feat_df[c].dtype == pl.Int8]
    float_cols = [c for c in feat_df.columns if c not in int8_cols]
    feat_df = feat_df.with_columns(
        [pl.col(c).cast(pl.Float32) for c in float_cols]
    )

    out = pl.concat([pairs, feat_df], how="horizontal")

    if emb is not None:
        # row-wise cosine (embeddings already L2-normalized -> dot product),
        # vectorized in chunks in the main process
        cos = np.empty(n, dtype=np.float32)
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            qv = emb[s1_arr[i:j]].astype(np.float32)
            cv = emb[c_arr[i:j]].astype(np.float32)
            cos[i:j] = np.sum(qv * cv, axis=1)
        out = out.with_columns(pl.Series("emb_cos", cos))

    return out


def add_rank_features(df, score_cols):
    """Per-s1 and per-c_idx rank features for each score col, plus
    candidate-count features. Callable twice: once on raw sims + a cheap
    combined score, again later on a 2nd-stage model probability.
    """
    exprs = []
    for s in score_cols:
        sfill = pl.col(s).fill_null(-1.0).fill_nan(-1.0)
        exprs += [
            sfill.rank(method="ordinal", descending=True).over("s1_idx").alias(f"{s}_rank_s1"),
            sfill.rank(method="ordinal", descending=True).over("c_idx").alias(f"{s}_rank_c"),
            (sfill.max().over("s1_idx") - sfill).alias(f"{s}_gap_s1max"),
            (sfill.max().over("c_idx") - sfill).alias(f"{s}_gap_cmax"),
        ]
    df = df.with_columns(exprs)

    mutual_exprs = [
        ((pl.col(f"{s}_rank_s1") == 1) & (pl.col(f"{s}_rank_c") == 1))
        .cast(pl.Int8)
        .alias(f"{s}_mutual_top1")
        for s in score_cols
    ]
    df = df.with_columns(mutual_exprs)

    df = df.with_columns(
        pl.len().over("s1_idx").alias("n_cands_s1"),
        pl.len().over("c_idx").alias("n_cands_c"),
    )
    return df


def add_group_features(df):
    """Duplicate-address / duplicate-name counts per s1 and per candidate.
    Needs addr_tset, name_tset columns (from build_features)."""
    a = (pl.col("addr_tset") >= 0.9).cast(pl.Float32)
    n = (pl.col("name_tset") >= 0.9).cast(pl.Float32)
    return df.with_columns(
        a.sum().over("s1_idx").alias("n_same_addr_s1"),
        n.sum().over("s1_idx").alias("n_same_name_s1"),
        a.sum().over("c_idx").alias("n_same_addr_c"),
    )


def _stage2_chunk(args):
    c_arr, a_arr = args
    g = _G
    nm, ad, nk = g["name"], g["addr"], g["name_key"]
    out = np.empty((len(c_arr), 3), np.float32)
    for k, (ci, ai) in enumerate(zip(c_arr.tolist(), a_arr.tolist())):
        out[k, 0] = fuzz.token_set_ratio(nm[ci], nm[ai])
        out[k, 1] = fuzz.token_set_ratio(ad[ci], ad[ai])
        out[k, 2] = fuzz.ratio(nk[ci], nk[ai])
    return out / 100.0


def add_stage2_features(df, prep, p_col="p", n_anchor=2, n_jobs=64, chunk=500_000):
    """Cluster consistency: anchors = top n_anchor candidates per s1 by p (p>=0.5).
    For every pair whose c is not itself the anchor, max over anchors of name/addr/
    name_key similarity to the anchor + anchor_p_max + n_anchors. NaN if no anchor.
    Also adds p rank features (add_rank_features on p_col)."""
    _set_globals(prep)
    anchors = (
        df.filter(pl.col(p_col) >= 0.5)
        .with_columns(pl.col(p_col).rank(method="ordinal", descending=True).over("s1_idx").alias("_ar"))
        .filter(pl.col("_ar") <= n_anchor)
        .select("s1_idx", pl.col("c_idx").alias("_anchor"), pl.col(p_col).alias("_ap"))
    )
    j = df.select("s1_idx", "c_idx").join(anchors, on="s1_idx").filter(pl.col("c_idx") != pl.col("_anchor"))
    c_arr, a_arr = j["c_idx"].to_numpy(), j["_anchor"].to_numpy()
    chunks = [(c_arr[i : i + chunk], a_arr[i : i + chunk]) for i in range(0, len(c_arr), chunk)]
    res = _pmap(_stage2_chunk, chunks, n_jobs)
    sims = np.concatenate(res) if res else np.empty((0, 3), np.float32)
    j = j.with_columns(
        pl.Series("s2_name_tset", sims[:, 0]), pl.Series("s2_addr_tset", sims[:, 1]),
        pl.Series("s2_namekey_ratio", sims[:, 2]),
    )
    agg = j.group_by("s1_idx", "c_idx").agg(
        pl.col("s2_name_tset").max(), pl.col("s2_addr_tset").max(), pl.col("s2_namekey_ratio").max(),
        pl.col("_ap").max().cast(pl.Float32).alias("anchor_p_max"),
        pl.len().cast(pl.Int8).alias("n_anchors"),
    )
    df = df.join(agg, on=["s1_idx", "c_idx"], how="left").with_columns(
        pl.col("n_anchors").fill_null(0).cast(pl.Int8),
        pl.col("s2_name_tset", "s2_addr_tset", "s2_namekey_ratio", "anchor_p_max").cast(pl.Float32).fill_null(np.nan),
    )
    return add_rank_features(df, [p_col])


def label_pairs(df, gt):
    """Add int8 label column: 1 if (s1_idx, c_idx) is a true match per gt."""
    gt_keys = gt.select("s1_idx", "c_idx").unique().with_columns(
        pl.lit(1, dtype=pl.Int8).alias("label")
    )
    out = df.join(gt_keys, on=["s1_idx", "c_idx"], how="left")
    out = out.with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    return out


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    import string

    random.seed(1)
    np.random.seed(1)

    def rand_word():
        return "".join(random.choices(string.ascii_lowercase, k=6))

    n = 200
    countries = ["US", "IN"]
    rows = []
    for i in range(n):
        c = random.choice(countries)
        nm = rand_word() + " " + rand_word()
        addr = rand_word() + " street " + str(random.randint(10000, 99999))
        rows.append(
            dict(
                idx=i,
                entity_id=f"e{i}",
                src=1 if i < 40 else random.choice([2, 3]),
                country=c,
                name_orig=nm,
                addr_orig=addr,
                name=nm,
                addr=addr,
                core=nm.split(" ")[0],
                legal=random.choice(["", "llc", "pvt ltd"]),
                name_key=nm[:4],
                addr_key=addr[:4],
                postcode=addr.split(" ")[-1],
                nums=addr.split(" ")[-1],
                script="latin",
                landmark="" if random.random() > 0.3 else "near park",
            )
        )
    prep = pl.DataFrame(rows).with_columns(pl.col("idx").cast(pl.Int32))

    # fabricate candidate pairs: each query vs 5 random pool rows of same country
    pool_by_country = {
        c: prep.filter((pl.col("country") == c) & (pl.col("src") != 1))["idx"].to_list()
        for c in countries
    }
    pair_rows = []
    for qrow in prep.filter(pl.col("src") == 1).iter_rows(named=True):
        candidates = pool_by_country[qrow["country"]]
        picks = random.sample(candidates, min(5, len(candidates)))
        for ci in picks:
            pair_rows.append(dict(s1_idx=qrow["idx"], c_idx=ci))
    pairs = pl.DataFrame(pair_rows).with_columns(
        pl.col("s1_idx", "c_idx").cast(pl.Int32)
    )

    feats = build_features(pairs, prep, emb=None, n_jobs=2, chunk=50)
    print("features shape:", feats.shape)
    assert feats.height == pairs.height
    assert "core_idf_jaccard" in feats.columns
    assert "country" not in feats.columns

    ranked = add_rank_features(feats, ["name_ratio", "core_ratio"])
    print("ranked cols added:", [c for c in ranked.columns if "rank" in c or "n_cands" in c][:6])
    assert "name_ratio_rank_s1" in ranked.columns
    assert "n_cands_s1" in ranked.columns

    for c in ("house_num_eq", "num_overlap_any", "addr_tset_x_num", "q_core_max_idf", "c_core_mean_idf",
              "shared_rare_token", "c_name_ntok", "q_addr_empty", "c_name_is_single_token_oov"):
        assert c in feats.columns, c
    # leading-zero normalisation
    p2 = prep.head(2).with_columns(pl.Series("nums", ["04348", "4348"]), pl.lit("US").alias("country"))
    f2 = build_features(pl.DataFrame({"s1_idx": [0], "c_idx": [1]}, schema={"s1_idx": pl.Int32, "c_idx": pl.Int32}),
                        p2, n_jobs=1)
    assert f2["house_num_eq"][0] == 1 and f2["num_overlap_any"][0] == 1

    g = add_group_features(feats)
    assert {"n_same_addr_s1", "n_same_name_s1", "n_same_addr_c"} <= set(g.columns)

    pp = np.random.RandomState(0).rand(feats.height)
    s2 = add_stage2_features(feats.with_columns(pl.Series("p", pp)), prep, n_jobs=2, chunk=50)
    assert s2.height == feats.height and "p_rank_s1" in s2.columns and "n_anchors" in s2.columns
    assert s2["n_anchors"].max() >= 1 and s2["s2_name_tset"].is_nan().sum() > 0
    print("stage2 ok, anchored:", (s2["n_anchors"] > 0).sum())

    # fabricate ground truth: first candidate of each query is "true"
    gt = pairs.group_by("s1_idx", maintain_order=True).first()
    labeled = label_pairs(ranked, gt)
    assert labeled["label"].sum() >= 1
    print("labeled positives:", labeled["label"].sum(), "/", labeled.height)

    print("features.py smoke test OK")
