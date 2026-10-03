"""Pipeline CLI. Each stage reads/writes parquet in $BER_WORK (see CONTRACT.md).

    prep:train prep:test folds embed:* block:* blockgpu:*      text views, folds (va/ce/tr), candidate search
    features:train ce_data cefit train1 cescore:train train    stage-0 cap, features, cross-encoder, stage 1/2
    features:test predict1:test cescore:test predict:test      test inference -> $BER_OUT/*.tsv
  (pbs/pipeline.pbs runs these in order; `cefit`/`cescore` are `python -m ber.ce train|score`)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(os.environ.get("BER_DATA", Path.home() / "amazonml/amazon-ml/dataset"))
WORK = Path(os.environ.get("BER_WORK", Path(__file__).resolve().parents[2] / "working"))
OUT = Path(os.environ.get("BER_OUT", Path(__file__).resolve().parents[2] / "output"))
NJ = int(os.environ.get("NJOBS", os.cpu_count()))
EMB_MODEL = os.environ.get("BER_EMB", "sentence-transformers/LaBSE")
EMB_TAG = EMB_MODEL.split("/")[-1]
EMB_PREFIX = "query: " if "e5" in EMB_MODEL else ""
N_VA, N_CE = int(os.environ.get("N_VA", 100_000)), int(os.environ.get("N_CE", 500_000))
FIT_S1 = int(os.environ.get("FIT_S1", 350_000))  # stage-1 fits on this many tr S1 per OOF half; predicts all
N_CHUNKS = int(os.environ.get("N_CHUNKS", 8))
K_TFIDF, K_NAME, K_EMB = 40, int(os.environ.get("K_NAME", 60)), 40
K_TOTAL, K_PER = int(os.environ.get("K_TOTAL", 60)), int(os.environ.get("K_PER", 10))
LGB_ROUNDS = int(os.environ.get("LGB_ROUNDS", 3000))
LGB_PARAMS = {"learning_rate": float(os.environ.get("LGB_LR", 0.05))}


def sim_cols(df):
    return [c for c in df.columns if c.endswith("_sim")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_tsv(p):
    # quote_char=None: names contain stray quotes; all columns as strings
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, empty_string_is_null=False)


def load_prep(split):
    return pl.read_parquet(WORK / f"{split}_prep.parquet")


def query_mask(prep, split):
    """Boolean mask over prep rows: which S1 get blocked. All S1 in both splits (see stage_folds)."""
    return (prep["src"] == 1).to_numpy()


def stage_prep(split):
    from ber.text import prep_frame
    frames = []
    for s in (1, 2, 3):
        df = read_tsv(DATA / split / f"{split}_source{s}.tsv").with_columns(pl.lit(s, pl.Int8).alias("src"))
        frames.append(df)
    raw = pl.concat(frames)
    log(split, "rows", raw.height)
    prep = prep_frame(raw, n_jobs=NJ).with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    prep.write_parquet(WORK / f"{split}_prep.parquet")
    log("wrote prep", prep.shape)
    if split != "train":
        return
    gt = read_tsv(DATA / "train/train_ground_truth.tsv")
    gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True)
            .with_columns(pl.col("matched_entity_ids").str.strip_chars('" ')))
    key = prep.select("entity_id", "idx")
    gt = (gt.join(key, left_on="source1_entity_id", right_on="entity_id")
            .join(key, left_on="matched_entity_ids", right_on="entity_id", suffix="_c")
            .select(pl.col("idx").alias("s1_idx"), pl.col("idx_c").alias("c_idx")))
    gt.write_parquet(WORK / "train_gt.parquet")
    log("gt pairs", gt.height)


def stage_sample(n):
    """Small real-data replica in $BER_WORK/sample/{train,test}: n S1 each, their true matches, plus
    distractors (~same ratio as the full pool). Test is built from *train* S1 so it can be scored."""
    import shutil
    out = WORK / "sample"
    gt = read_tsv(DATA / "train/train_ground_truth.tsv")
    rng = np.random.default_rng(1)
    pick = rng.permutation(gt.height)[: 2 * n]
    s = [read_tsv(DATA / f"train/train_source{i}.tsv") for i in (1, 2, 3)]
    for split, rows in (("train", pick[:n]), ("test", pick[n:])):
        g = gt[rows]
        (out / split).mkdir(parents=True, exist_ok=True)
        m = set(x for v in g["matched_entity_ids"] for x in v.split(",") if x)
        s[0].filter(pl.col("entity_id").is_in(g["source1_entity_id"])).write_csv(out / split / f"{split}_source1.tsv", separator="\t", quote_style="never")
        for i in (2, 3):
            pool = s[i - 1]
            hit = pool.filter(pl.col("entity_id").is_in(list(m)))
            noise = pool.sample(int(hit.height * 0.5), seed=i)
            pl.concat([hit, noise]).unique("entity_id").write_csv(out / split / f"{split}_source{i}.tsv", separator="\t", quote_style="never")
        g.write_csv(out / split / f"{split}_ground_truth.tsv", separator="\t", quote_style="never")
    log("sample written", out)


def stage_folds():
    """va = first N_VA of a fixed S1 permutation (same val across versions), ce = next N_CE (cross-encoder fit,
    disjoint from tr/va), tr = rest. Blocking/features run on ALL train S1 so each pool record sees the same
    S1 competition as in test (blocking only fold S1 gave ~5 S1 per pool record vs ~22 in test)."""
    prep = load_prep("train")
    s1 = prep.filter(pl.col("src") == 1)["idx"].to_numpy()
    perm = np.random.default_rng(0).permutation(s1)
    fold = np.array(["tr"] * len(perm), dtype=object)
    fold[:N_VA], fold[N_VA:N_VA + N_CE] = "va", "ce"
    pl.DataFrame({"s1_idx": perm, "fold": fold.astype(str)}, schema={"s1_idx": pl.Int32, "fold": pl.Utf8}) \
        .write_parquet(WORK / "train_folds.parquet")
    log("folds", {f: int((fold == f).sum()) for f in ("va", "ce", "tr")})


def scan_feats(split):
    return pl.scan_parquet(WORK / f"{split}_feats_*.parquet")


def stage_ce_data():
    """Cross-encoder fit pairs from the 'ce' fold: all positives + hard negatives (top-10 by stage-0 p0)."""
    ce_s1 = pl.read_parquet(WORK / "train_folds.parquet").filter(pl.col("fold") == "ce")["s1_idx"]
    df = (scan_feats("train").filter(pl.col("s1_idx").is_in(ce_s1.implode()))
          .select("s1_idx", "c_idx", "label", "p0").collect()
          .filter((pl.col("label") == 1) | (pl.col("p0").rank("ordinal", descending=True).over("s1_idx") <= 10)))
    n = int(os.environ.get("CE_MAX", 6_000_000))
    df = df.sample(min(n, df.height), seed=0).select("s1_idx", "c_idx", "label")
    df.write_parquet(WORK / "ce_fit_pairs.parquet")
    log("ce fit pairs", df.height, "pos", df["label"].sum())


def stage_embed(split):
    from ber.block import emb_encode
    prep = load_prep(split)
    texts = (pl.col("name_orig") + " | " + pl.col("addr_orig"))
    texts = prep.select(texts.alias("t"))["t"].to_list()
    emb = emb_encode(texts, EMB_MODEL, prefix=EMB_PREFIX)
    np.save(WORK / f"{split}_emb_{EMB_TAG}.npy", emb)
    log("emb", emb.shape)


def stage_block(split, gpu):
    from ber import block
    prep = load_prep(split)
    qm = query_mask(prep, split)

    def run(fn, m, *args, **kw):
        """Append mode: S1 already blocked in an existing file (e.g. copied from a previous version) are skipped."""
        f = WORK / f"{split}_blk_{m}.parquet"
        old = pl.read_parquet(f) if f.exists() else None
        q = qm.copy()
        if old is not None:
            q[old["s1_idx"].unique().to_numpy()] = False
            if not q.any():
                log(m, "already complete")
                return
        new = fn(*args, q_mask=q, **kw)
        (pl.concat([old, new], how="diagonal_relaxed") if old is not None else new).write_parquet(f)
        log(m, "done, new S1", int(q.sum()))

    if gpu:
        emb = np.load(WORK / f"{split}_emb_{EMB_TAG}.npy", mmap_mode="r")
        run(block.emb_knn, "emb", emb, prep, K_EMB)
        return
    run(block.tfidf_knn, "tfidf", prep, ["name", "addr"], K_TFIDF, n_jobs=NJ)
    run(block.tfidf_knn, "name", prep, ["core"], K_NAME, n_jobs=NJ, name="tfidf_name_sim")
    run(block.tfidf_knn, "skel", prep, ["name_key"], K_NAME, n_jobs=NJ, name="tfidf_key_sim", ngram_range=(2, 3))
    run(block.tfidf_knn, "coarse", prep, ["name_key2"], K_NAME, n_jobs=NJ, name="tfidf_coarse_sim", ngram_range=(2, 4))
    run(block.tfidf_knn, "addr", prep, ["addr"], 20, n_jobs=NJ, name="tfidf_addr_sim")
    run(block.key_block, "key", prep)


def stage_features(split):
    """Union -> learned cap -> global competition ranks (over ALL S1) -> expensive pair features in S1 chunks."""
    from ber import block, features
    prep = load_prep(split)
    files = [WORK / f"{split}_blk_{m}.parquet" for m in BLK_METHODS if (WORK / f"{split}_blk_{m}.parquet").exists()]
    folds = pl.read_parquet(WORK / "train_folds.parquet") if split == "train" else None
    capped = []
    for i in range(N_CHUNKS):  # union + cap per S1 chunk (a full 7-way outer join over 2.2M S1 is ~480M rows)
        ch = pl.col("s1_idx").hash(11) % N_CHUNKS == i
        pairs = block.union_pairs(*[pl.scan_parquet(f).filter(ch).collect() for f in files])
        n_union = pairs.height
        if split == "train" and i == 0:  # stage-0 ranker is fit once, on chunk 0's tr S1
            tr0 = folds.filter((pl.col("fold") == "tr") & ch)["s1_idx"].to_numpy()
            pairs = block.learned_cap(pairs, K_TOTAL, WORK / "lgb0.txt", gt=pl.read_parquet(WORK / "train_gt.parquet"),
                                      train_s1=tr0, per_method=K_PER)
        else:
            pairs = block.learned_cap(pairs, K_TOTAL, WORK / "lgb0.txt", per_method=K_PER)
        capped.append(pairs)
        log("chunk", i, "union", n_union, "capped", pairs.height)
    pairs = pl.concat(capped, how="diagonal_relaxed")
    del capped
    log("capped pairs", pairs.height, "methods", sim_cols(pairs))
    if split == "train":
        gt = pl.read_parquet(WORK / "train_gt.parquet")
        va = folds.filter(pl.col("fold") == "va")["s1_idx"].to_numpy()
        rep = block.recall_report(pairs, gt, va)
        (WORK / "blocking_report.json").write_text(json.dumps(rep, indent=2, default=float))
        log("blocking (val S1)", {k: v for k, v in rep.items() if k != "recall_at_k"})
    pairs = features.add_rank_features(pairs, sim_cols(pairs))  # competition over every S1, as in test
    emb_path = WORK / f"{split}_emb_{EMB_TAG}.npy"
    emb = np.load(emb_path, mmap_mode="r") if emb_path.exists() else None
    chunk = (pairs["s1_idx"].hash(11) % N_CHUNKS).to_numpy()
    gt = pl.read_parquet(WORK / "train_gt.parquet") if split == "train" else None
    for i in range(N_CHUNKS):
        df = features.build_features(pairs.filter(pl.Series(chunk == i)), prep, emb, n_jobs=NJ)
        a, n = (pl.col("addr_tset") >= 0.9).cast(pl.Int16), (pl.col("name_tset") >= 0.9).cast(pl.Int16)
        df = df.with_columns(a.sum().over("s1_idx").alias("n_same_addr_s1"), n.sum().over("s1_idx").alias("n_same_name_s1"))
        if gt is not None:
            df = features.label_pairs(df, gt)
        df.write_parquet(WORK / f"{split}_feats_{i}.parquet")
        log("features chunk", i, df.shape)


BLK_METHODS = ["tfidf", "name", "skel", "coarse", "addr", "emb", "key"]  # explicit: no stray globbed backups
NON_FEAT = ("s1_idx", "c_idx", "label", "fold", "p", "p1", "p0", "g")


def feature_cols(names):
    return [c for c in names if c not in NON_FEAT]


def fit(tr, va, cols):
    return __import__("ber.model", fromlist=["x"]).train_lgb(
        tr.select(cols).to_numpy(), tr["label"].to_numpy(), va.select(cols).to_numpy(), va["label"].to_numpy(),
        params=LGB_PARAMS, num_rounds=LGB_ROUNDS)


CE_BAND = float(os.environ.get("CE_BAND", 0.005))
S2_MIN = float(os.environ.get("S2_MIN", 0.001))  # stage-2 sees only pairs stage 1 didn't rule out
STAGE1_ONLY = False


def add_ce(df, split):
    """Join cross-encoder score if it was computed (pairs below CE_BAND stay null)."""
    f = WORK / f"ce_{split}_scores.parquet"
    return df.join(pl.read_parquet(f), on=["s1_idx", "c_idx"], how="left") if f.exists() else df


def stage1_predict_all(split, cols):
    """Stage-1 p for EVERY pair. train: tr rows get the out-of-fold model, other folds the fold average."""
    from ber import model
    import lightgbm as lgb
    bsts = [lgb.Booster(model_file=str(WORK / f"lgb1_{i}.txt")) for i in (0, 1)]
    tr_s1 = None
    if split == "train":
        tr_s1 = pl.read_parquet(WORK / "train_folds.parquet").filter(pl.col("fold") == "tr")["s1_idx"]
    out = []
    for f in sorted(WORK.glob(f"{split}_feats_*.parquet")):
        d = pl.read_parquet(f, columns=["s1_idx", "c_idx"] + cols)
        X = d.select(cols).to_numpy()
        pa, pb = model.predict(bsts[0], X), model.predict(bsts[1], X)
        del X
        p = (pa + pb) / 2
        if tr_s1 is not None:
            g = (d["s1_idx"].hash(7) % 2).to_numpy()
            istr = d["s1_idx"].is_in(tr_s1.implode()).to_numpy()
            p = np.where(istr & (g == 0), pb, np.where(istr & (g == 1), pa, p))  # model k was fit on half k
        out.append(d.select("s1_idx", "c_idx").with_columns(pl.Series("p", p.astype(np.float32))))
        log("stage1 predicted", f.name)
    pl.concat(out).write_parquet(WORK / f"p1_{split}.parquet")


def stage2_rows(split):
    """All pairs (every S1) that stage 1 didn't rule out, with features + CE + stage-2 features."""
    from ber import features
    keep = pl.read_parquet(WORK / f"p1_{split}.parquet").filter(pl.col("p") >= S2_MIN)
    df = scan_feats(split).join(keep.lazy(), on=["s1_idx", "c_idx"], how="inner").collect()
    return features.add_stage2_features(add_ce(df, split), load_prep(split), p_col="p", n_jobs=NJ).rename({"p": "p1"})


def evaluate_all(df, gt_va, s1_va, tag):
    """1-to-1 argmax over ALL S1 (val S1 compete with every train S1, like test), then tune on val S1."""
    from ber import model
    am = df.select("s1_idx", "c_idx", "p").sort("p", descending=True).group_by("c_idx").first()
    va = am.filter(pl.col("s1_idx").is_in(s1_va))
    t, a, score, table = model.tune(va, gt_va, s1_va)
    pred = model.assign(va, t, a)
    groups = load_prep("train").select(pl.col("idx").alias("s1_idx"), "country", "script").filter(pl.col("s1_idx").is_in(s1_va))
    by_country = model.macro_f05_by(pred, gt_va, s1_va, groups.select("s1_idx", pl.col("country").alias("group")))
    by_script = model.macro_f05_by(pred, gt_va, s1_va, groups.select("s1_idx", pl.col("script").alias("group")))
    log(tag, "VAL F0.5", round(score, 5), "t", t, "alpha", a, "\n", by_country)
    return {"t": t, "alpha": a, "val_f05": score, "by_country": by_country.to_dicts(),
            "by_script": by_script.to_dicts()}, table


def stage_train():
    from ber import model
    folds = pl.read_parquet(WORK / "train_folds.parquet")
    s1_va = folds.filter(pl.col("fold") == "va")["s1_idx"].to_numpy()
    gt_va = pl.read_parquet(WORK / "train_gt.parquet").filter(pl.col("s1_idx").is_in(s1_va))
    cols = feature_cols(scan_feats("train").collect_schema().names())
    if not (WORK / "p1_train.parquet").exists():
        # stage 1: two halves of tr (by S1 hash), each fit on <= FIT_S1 S1, early-stopped on val
        tr_s1 = folds.filter(pl.col("fold") == "tr").with_columns((pl.col("s1_idx").hash(7) % 2).alias("g"))
        va = scan_feats("train").filter(pl.col("s1_idx").is_in(pl.Series(s1_va).implode())).collect()
        for k in (0, 1):
            half = tr_s1.filter(pl.col("g") == k)["s1_idx"]
            s1k = half.sample(min(FIT_S1, half.len()), seed=k)
            tr = scan_feats("train").filter(pl.col("s1_idx").is_in(s1k.implode())).collect()
            log("stage1 half", k, "fit rows", tr.height, "pos", tr["label"].sum())
            fit(tr, va, cols).save_model(str(WORK / f"lgb1_{k}.txt"))
            del tr
        del va
        stage1_predict_all("train", cols)
    p1 = pl.read_parquet(WORK / "p1_train.parquet")
    res1, _ = evaluate_all(p1, gt_va, s1_va, "stage1")
    if STAGE1_ONLY:  # cross-encoder pairs for every S1 (competition needs every S1's stage-2 p)
        p1.filter(pl.col("p") >= CE_BAND).select("s1_idx", "c_idx").write_parquet(WORK / "ce_train_pairs.parquet")
        return
    df = stage2_rows("train").join(folds, on="s1_idx", how="left")
    cols2 = feature_cols(df.columns) + ["p1"]
    tr, va = df.filter(pl.col("fold") == "tr"), df.filter(pl.col("fold") == "va")
    log("stage2 rows all", df.height, "tr", tr.height, "va", va.height, "features", len(cols2), "ce" in cols2)
    bst2 = fit(tr, va, cols2)
    bst2.save_model(str(WORK / "lgb2.txt"))
    del tr, va
    df = df.with_columns(pl.Series("p", model.predict(bst2, df.select(cols2).to_numpy())))
    res, table = evaluate_all(df, gt_va, s1_va, "stage2")
    imp = dict(zip(cols2, bst2.feature_importance("gain").round(1).tolist()))
    res.update({"stage1": res1, "features": cols, "features2": cols2, "best_iter": bst2.best_iteration,
                "importance": dict(sorted(imp.items(), key=lambda x: -x[1]))})
    (WORK / "train_result.json").write_text(json.dumps(res, indent=2, default=float))
    table.write_csv(WORK / "tune_table.csv")
    df.filter(pl.col("s1_idx").is_in(s1_va)).select("s1_idx", "c_idx", "p1", "p", "label").write_parquet(WORK / "val_pred.parquet")


def stage_predict():
    import lightgbm as lgb
    from ber import model
    res = json.loads((WORK / "train_result.json").read_text())
    if not (WORK / "p1_test.parquet").exists():
        stage1_predict_all("test", res["features"])
        pl.read_parquet(WORK / "p1_test.parquet").filter(pl.col("p") >= CE_BAND).select("s1_idx", "c_idx") \
            .write_parquet(WORK / "ce_test_pairs.parquet")
    if STAGE1_ONLY:
        return
    df = stage2_rows("test")
    df = df.with_columns(pl.Series("p", model.predict(lgb.Booster(model_file=str(WORK / "lgb2.txt")),
                                                      df.select(res["features2"]).to_numpy())))
    pred = model.assign(df.select("s1_idx", "c_idx", "p"), res["t"], res["alpha"])
    cand = pl.read_parquet(WORK / "p1_test.parquet", columns=["s1_idx", "c_idx"])  # every pair the models scored
    prep = load_prep("test").select("idx", "entity_id", "src", "country")
    eid = prep["entity_id"]

    def to_ids(d):
        return pl.DataFrame([eid.gather(d["s1_idx"]).alias("s1_id"), eid.gather(d["c_idx"]).alias("c_id")])
    OUT.mkdir(parents=True, exist_ok=True)
    s1_ids = prep.filter(pl.col("src") == 1)["entity_id"].to_list()
    model.write_submission(s1_ids, to_ids(pred), OUT / "matching_results.tsv")
    model.write_submission(s1_ids, to_ids(cand), OUT / "candidate_pairs.tsv", col="candidate_entity_ids")
    issues = model.validate(OUT / "matching_results.tsv", OUT / "candidate_pairs.tsv", set(s1_ids),
                            set(prep.filter(pl.col("src") != 1)["entity_id"].to_list()))
    stats = (pred.join(prep.rename({"idx": "s1_idx"}), on="s1_idx").group_by("country")
             .agg(pl.len().alias("pairs"), pl.col("s1_idx").n_unique().alias("s1_matched")))
    log("test matches by country\n", stats)
    log("VALIDATION:", "PASS" if not issues else issues[:20])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["folds", "ce_data", "train1", "predict1", "sample", "prep", "embed", "block", "features", "train", "predict"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n", type=int, default=20_000)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    log("stage", a.stage, a.split, "work", WORK)
    if a.stage in ("train1", "predict1"):
        STAGE1_ONLY = True
        a.stage = a.stage[:-1]
    {"folds": stage_folds, "ce_data": stage_ce_data, "sample": lambda: stage_sample(a.n), "prep": lambda: stage_prep(a.split), "embed": lambda: stage_embed(a.split),
     "block": lambda: stage_block(a.split, a.gpu), "features": lambda: stage_features(a.split),
     "train": stage_train, "predict": stage_predict}[a.stage]()
    log("done")
