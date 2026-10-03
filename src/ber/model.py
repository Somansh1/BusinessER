"""
Matching, tuning, scoring, LightGBM training, submission I/O and validation
for the entity-resolution task described in CONTRACT.md.

Everything is vectorized (polars group_by / joins) - no python loops over
entities. Designed to run at 2.2M S1 x up to 70M candidate pairs.
"""

import os
import csv
import tempfile

import numpy as np
import polars as pl
import lightgbm as lgb


# ---------------------------------------------------------------------------
# 1. scoring
# ---------------------------------------------------------------------------

def _f05_from_counts(tp: pl.Series, fp: pl.Series, fn: pl.Series) -> pl.Series:
    # F0.5 = 1.25 P R / (0.25 P + R) = 1.25 tp / (1.25 tp + fp + 0.25 fn)
    denom = 1.25 * tp + fp + 0.25 * fn
    return pl.when(denom == 0).then(0.0).otherwise(1.25 * tp / denom)


def _per_s1_scores(pred: pl.DataFrame, truth: pl.DataFrame, s1_all: np.ndarray) -> pl.DataFrame:
    """Return pl.DataFrame[s1_idx, score] for every s1 in s1_all."""
    pred = pred.select(["s1_idx", "c_idx"]).unique()
    truth = truth.select(["s1_idx", "c_idx"]).unique()

    n_pred = pred.group_by("s1_idx").agg(pl.len().alias("n_pred"))
    n_truth = truth.group_by("s1_idx").agg(pl.len().alias("n_truth"))

    tp = (
        pred.join(truth, on=["s1_idx", "c_idx"], how="inner")
        .group_by("s1_idx")
        .agg(pl.len().alias("tp"))
    )

    base = pl.DataFrame({"s1_idx": np.asarray(s1_all, dtype=np.int32)})
    scored = (
        base.join(n_pred, on="s1_idx", how="left")
        .join(n_truth, on="s1_idx", how="left")
        .join(tp, on="s1_idx", how="left")
        .with_columns(
            pl.col("n_pred").fill_null(0),
            pl.col("n_truth").fill_null(0),
            pl.col("tp").fill_null(0),
        )
        .with_columns(
            (pl.col("n_pred") - pl.col("tp")).alias("fp"),
            (pl.col("n_truth") - pl.col("tp")).alias("fn"),
        )
        .with_columns(
            pl.when((pl.col("n_truth") == 0) & (pl.col("n_pred") == 0))
            .then(1.0)
            .when((pl.col("n_truth") == 0) & (pl.col("n_pred") > 0))
            .then(0.0)
            .when((pl.col("n_truth") > 0) & (pl.col("n_pred") == 0))
            .then(0.0)
            .otherwise(_f05_from_counts(pl.col("tp"), pl.col("fp"), pl.col("fn")))
            .alias("score")
        )
    )
    return scored.select(["s1_idx", "score"])


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_all: np.ndarray) -> float:
    """Macro-averaged F0.5 over ALL s1 in s1_all (pred/truth: cols s1_idx,c_idx)."""
    scored = _per_s1_scores(pred, truth, s1_all)
    return float(scored["score"].mean())


def macro_f05_by(pred: pl.DataFrame, truth: pl.DataFrame, s1_all: np.ndarray, groups) -> pl.DataFrame:
    """Per-group macro F0.5. `groups` is dict s1_idx->label or pl.DataFrame[s1_idx,group]."""
    scored = _per_s1_scores(pred, truth, s1_all)
    if isinstance(groups, dict):
        g = pl.DataFrame({
            "s1_idx": np.array(list(groups.keys()), dtype=np.int32),
            "group": list(groups.values()),
        })
    else:
        g = groups.rename({groups.columns[1]: "group"}) if groups.columns[1] != "group" else groups
    return (
        scored.join(g, on="s1_idx", how="left")
        .group_by("group")
        .agg(pl.col("score").mean().alias("macro_f05"), pl.len().alias("n"))
        .sort("group")
    )


# ---------------------------------------------------------------------------
# 2. 1-to-1 assignment
# ---------------------------------------------------------------------------

def assign(pairs: pl.DataFrame, t: float, alpha: float) -> pl.DataFrame:
    """
    pairs: s1_idx, c_idx, p.
    Step 1: each c_idx keeps only its argmax-p s1 (1-to-1 on the S2/S3 side).
    Step 2: keep p >= t and p >= alpha * max_p for that s1.
    """
    best_per_c = (
        pairs.sort("p", descending=True)
        .group_by("c_idx", maintain_order=False)
        .first()
    )
    max_p_per_s1 = best_per_c.group_by("s1_idx").agg(pl.col("p").max().alias("max_p"))
    out = (
        best_per_c.join(max_p_per_s1, on="s1_idx", how="left")
        .filter((pl.col("p") >= t) & (pl.col("p") >= alpha * pl.col("max_p")))
        .select(["s1_idx", "c_idx"])
    )
    return out


# ---------------------------------------------------------------------------
# 3. threshold / alpha tuning
# ---------------------------------------------------------------------------

def tune(pairs_val_with_p, truth_val, s1_val_all,
         ts=np.arange(0.2, 0.95, 0.025), alphas=(0, 0.3, 0.5, 0.7)):
    """Grid search (t, alpha) maximizing macro_f05. Precomputes the argmax-per-c
    filter once (it doesn't depend on t/alpha), so the grid loop is cheap."""
    best_per_c = (
        pairs_val_with_p.sort("p", descending=True)
        .group_by("c_idx", maintain_order=False)
        .first()
    )
    max_p_per_s1 = best_per_c.group_by("s1_idx").agg(pl.col("p").max().alias("max_p"))
    base = best_per_c.join(max_p_per_s1, on="s1_idx", how="left")

    rows = []
    best_t, best_alpha, best_score = None, None, -1.0
    for alpha in alphas:
        cand = base.filter(pl.col("p") >= alpha * pl.col("max_p"))
        for t in ts:
            pred = cand.filter(pl.col("p") >= t).select(["s1_idx", "c_idx"])
            score = macro_f05(pred, truth_val, s1_val_all)
            rows.append({"t": float(t), "alpha": float(alpha), "score": score})
            if score > best_score:
                best_t, best_alpha, best_score = float(t), float(alpha), score

    table = pl.DataFrame(rows)
    return best_t, best_alpha, best_score, table


# ---------------------------------------------------------------------------
# 4. LightGBM
# ---------------------------------------------------------------------------

def train_lgb(X_tr, y_tr, X_va, y_va, params=None, num_rounds=3000):
    default_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", 64)),
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        default_params,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return booster


def predict(booster, X) -> np.ndarray:
    return booster.predict(X, num_iteration=booster.best_iteration)


# ---------------------------------------------------------------------------
# 5. submission writing
# ---------------------------------------------------------------------------

def write_submission(s1_ids, matches: pl.DataFrame, path, col="matched_entity_ids"):
    """matches: pl.DataFrame[s1_id, c_id]. One TSV row per s1 in s1_ids, comma-joined
    deduped c_ids (empty string if none)."""
    id_col = matches.columns[0]
    val_col = matches.columns[1]
    grouped = (
        matches.unique()
        .group_by(id_col)
        .agg(pl.col(val_col).unique().alias("vals"))
    )
    lut = dict(zip(grouped[id_col].to_list(), grouped["vals"].to_list()))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writerow(["source1_entity_id", col])
        for sid in s1_ids:
            vals = lut.get(sid, [])
            w.writerow([sid, ",".join(vals)])


# ---------------------------------------------------------------------------
# 6. validation
# ---------------------------------------------------------------------------

def _read_tsv(path):
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        rows = list(r)
    return rows


def validate(matching_path, candidate_path, test_s1_ids: set, test_pool_ids: set) -> list:
    issues = []

    m_rows = _read_tsv(matching_path)
    c_rows = _read_tsv(candidate_path)

    if not m_rows or m_rows[0] != ["source1_entity_id", "matched_entity_ids"]:
        issues.append("matching_path: bad header")
    if not c_rows or c_rows[0] != ["source1_entity_id", "candidate_entity_ids"]:
        issues.append("candidate_path: bad header")

    def parse(rows, col_name):
        seen_s1 = set()
        dups = set()
        data = {}
        for row in rows[1:]:
            if len(row) != 2:
                issues.append(f"{col_name}: malformed row {row}")
                continue
            sid, val = row
            if sid in seen_s1:
                dups.add(sid)
            seen_s1.add(sid)
            ids = [x for x in val.split(",") if x != ""] if val else []
            data[sid] = ids
        if dups:
            issues.append(f"{col_name}: duplicate source1 rows: {sorted(dups)[:10]}")
        return data

    m_data = parse(m_rows, "matching")
    c_data = parse(c_rows, "candidate")

    # every test s1 exactly once
    missing_m = test_s1_ids - set(m_data.keys())
    if missing_m:
        issues.append(f"matching: missing S1 ids: {sorted(missing_m)[:10]}")
    extra_m = set(m_data.keys()) - test_s1_ids
    if extra_m:
        issues.append(f"matching: unknown S1 ids: {sorted(extra_m)[:10]}")

    missing_c = test_s1_ids - set(c_data.keys())
    if missing_c:
        issues.append(f"candidate: missing S1 ids: {sorted(missing_c)[:10]}")

    def check_ids(data, col_name):
        for sid, ids in data.items():
            if len(ids) != len(set(ids)):
                issues.append(f"{col_name}: duplicate ids in row {sid}")
            for cid in ids:
                if not (cid.startswith("S2-") or cid.startswith("S3-")):
                    issues.append(f"{col_name}: bad id prefix '{cid}' in row {sid}")
                elif cid not in test_pool_ids:
                    issues.append(f"{col_name}: id '{cid}' not in pool (row {sid})")

    check_ids(m_data, "matching")
    check_ids(c_data, "candidate")

    # matches subset of candidates, per row
    for sid, ids in m_data.items():
        cand_ids = set(c_data.get(sid, []))
        bad = [i for i in ids if i not in cand_ids]
        if bad:
            issues.append(f"matching: ids not in candidates for {sid}: {bad[:5]}")

    return issues


# ---------------------------------------------------------------------------
# 7. self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- macro_f05: PDF example, truth {b1,b3}, pred {b1,b2,b3} -> F0.5 = 0.714
    pred = pl.DataFrame({"s1_idx": [0, 0, 0], "c_idx": [1, 2, 3]})
    truth = pl.DataFrame({"s1_idx": [0, 0], "c_idx": [1, 3]})
    score = macro_f05(pred, truth, np.array([0]))
    assert abs(score - 0.714) < 1e-3, score

    # --- singleton with empty truth, predicted empty -> 1.0
    pred_empty = pl.DataFrame({"s1_idx": [], "c_idx": []}, schema={"s1_idx": pl.Int32, "c_idx": pl.Int32})
    truth_empty = pl.DataFrame({"s1_idx": [], "c_idx": []}, schema={"s1_idx": pl.Int32, "c_idx": pl.Int32})
    score = macro_f05(pred_empty, truth_empty, np.array([5]))
    assert abs(score - 1.0) < 1e-9, score

    # --- singleton with empty truth, predicted something -> 0.0
    pred_something = pl.DataFrame({"s1_idx": [5], "c_idx": [9]})
    score = macro_f05(pred_something, truth_empty, np.array([5]))
    assert abs(score - 0.0) < 1e-9, score

    # --- assign: enforce 1-to-1 (c_idx shared by two s1, higher p wins)
    pairs = pl.DataFrame({
        "s1_idx": [0, 1, 2],
        "c_idx": [10, 10, 11],
        "p": [0.9, 0.95, 0.4],
    })
    out = assign(pairs, t=0.3, alpha=0.0)
    assert out.filter(pl.col("c_idx") == 10).height == 1
    assert out.filter(pl.col("c_idx") == 10)["s1_idx"][0] == 1
    assert out.filter(pl.col("c_idx") == 11).height == 1

    # --- validate: catches a dup id and a missing S1
    with tempfile.TemporaryDirectory() as d:
        mpath = os.path.join(d, "matching.tsv")
        cpath = os.path.join(d, "candidates.tsv")
        with open(mpath, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-1,S2-1\n")  # dup id
            # S1-2 missing entirely
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-1,S2-2\n")
            f.write("S1-2\tS2-3\n")

        issues = validate(mpath, cpath, test_s1_ids={"S1-1", "S1-2"}, test_pool_ids={"S2-1", "S2-2", "S2-3"})
        assert any("duplicate ids" in i for i in issues), issues
        assert any("missing S1 ids" in i for i in issues), issues

    # --- validate: clean pass -> []
    with tempfile.TemporaryDirectory() as d:
        mpath = os.path.join(d, "matching.tsv")
        cpath = os.path.join(d, "candidates.tsv")
        with open(mpath, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            f.write("S1-1\tS2-1\n")
            f.write("S1-2\t\n")
        with open(cpath, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-1\tS2-1,S2-2\n")
            f.write("S1-2\t\n")
        issues = validate(mpath, cpath, test_s1_ids={"S1-1", "S1-2"}, test_pool_ids={"S2-1", "S2-2"})
        assert issues == [], issues

    # --- write_submission smoke test
    with tempfile.TemporaryDirectory() as d:
        outp = os.path.join(d, "sub.tsv")
        matches = pl.DataFrame({"s1_id": ["S1-1", "S1-1"], "c_id": ["S2-1", "S2-1"]})  # dup on purpose
        write_submission(["S1-1", "S1-2"], matches, outp, col="matched_entity_ids")
        with open(outp, encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert lines[1] == "S1-1\tS2-1"  # deduped
        assert lines[2] == "S1-2\t"

    print("all self-checks passed")
