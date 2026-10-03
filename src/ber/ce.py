"""Cross-encoder reranker feature (multilingual). Input pair text = both records' text.

    python -m ber.ce train --split train    (reads WORK/ce_fit_pairs.parquet s1_idx,c_idx,label -> WORK/ce_model)
    python -m ber.ce score --split test|train (reads WORK/ce_{split}_pairs.parquet -> WORK/ce_{split}_scores.parquet)
    python -m ber.ce selftest

Env: BER_CE (default xlm-roberta-base), HF_HUB_OFFLINE=1 on GPU nodes. Uses cuda:0 of CUDA_VISIBLE_DEVICES.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

CE_MODEL = os.environ.get("BER_CE", "FacebookAI/xlm-roberta-large")
MAXLEN = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_text(prep: pl.DataFrame) -> list:
    """Text per prep row (index = prep idx). Non-latin script: append romanized name."""
    nl = pl.col("script") != "latin"
    t = pl.when(nl).then(pl.col("name_orig") + " (" + pl.col("name") + ") | " + pl.col("addr_orig")) \
          .otherwise(pl.col("name_orig") + " | " + pl.col("addr_orig"))
    return prep.sort("idx").select(t.alias("t"))["t"].to_list()


def _dev():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _enc(tok, texts, s1, c, dev):
    b = tok([texts[i] for i in s1], [texts[i] for i in c], max_length=MAXLEN, truncation="longest_first",
            padding=True, return_tensors="pt")
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def train_ce(pairs, texts, out_dir, epochs=1, bs=256, lr=2e-5, warmup=0.05, max_pairs=4_000_000, seed=0,
             model_name=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    name = model_name or CE_MODEL
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    s1, c, y = (pairs[k].to_numpy() for k in ("s1_idx", "c_idx", "label"))
    if len(s1) > max_pairs:  # ponytail: uniform subsample; stratify/hard-negative pick if needed
        k = rng.choice(len(s1), max_pairs, replace=False)
        s1, c, y = s1[k], c[k], y[k]
    y = y.astype(np.float32)
    dev = _dev()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    spe = (len(s1) + bs - 1) // bs
    total = spe * epochs
    sch = get_linear_schedule_with_warmup(opt, int(warmup * total), total)
    lossf = torch.nn.BCEWithLogitsLoss()
    log("ce train", name, "pairs", len(s1), "steps", total, "dev", dev, "pos", float(y.mean()))
    step, t0, run, seen = 0, time.time(), 0.0, 0
    for ep in range(epochs):
        perm = rng.permutation(len(s1))
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            b = _enc(tok, texts, s1[j], c[j], dev)
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logit = model(**b).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(y[j]).to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run += loss.item() if step % 50 == 0 or step % 500 == 0 else 0.0
            seen += len(j)
            if step % 500 == 0:
                log(f"ep {ep} step {step}/{total} loss(last) {loss.item():.4f} {seen / (time.time() - t0):.0f} pairs/s")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log("saved", out_dir)


def score_ce(pairs, texts, model_dir, bs=1024):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = _dev()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(dev).eval()
    s1, c = pairs["s1_idx"].to_numpy(), pairs["c_idx"].to_numpy()
    n = len(s1)
    tl = np.fromiter((len(t) for t in texts), np.int32, len(texts))
    order = np.argsort(tl[s1] + tl[c], kind="stable")  # char length ~ token length; buckets tight batches
    out = np.zeros(n, np.float32)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
        for k, i in enumerate(range(0, n, bs)):
            j = order[i:i + bs]
            p = torch.sigmoid(model(**_enc(tok, texts, s1[j], c[j], dev)).logits.squeeze(-1).float())
            out[j] = p.cpu().numpy()
            if k % 200 == 0:
                log(f"score {i}/{n} {(i + len(j)) / (time.time() - t0):.0f} pairs/s")
    return out


def selftest():
    import tempfile
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CE_MODEL, local_files_only=True)
    except Exception as e:  # torch/transformers missing or model not cached
        print("selftest SKIP:", type(e).__name__, str(e)[:120])
        return
    d = Path(tempfile.mkdtemp())
    cfg = AutoConfig.for_model("xlm-roberta", vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                               num_attention_heads=2, intermediate_size=64, num_labels=1, max_position_embeddings=200)
    AutoModelForSequenceClassification.from_config(cfg).save_pretrained(d / "init")
    tok.save_pretrained(d / "init")
    rng = np.random.default_rng(0)
    prep = pl.DataFrame({"idx": np.arange(40, dtype=np.int32),
                         "name_orig": [f"acme {i} ltd" for i in range(40)], "addr_orig": [f"{i} main st" for i in range(40)],
                         "name": [f"acme {i} ltd" for i in range(40)], "script": ["latin", "tamil"] * 20})
    texts = build_text(prep)
    pairs = pl.DataFrame({"s1_idx": rng.integers(0, 40, 64), "c_idx": rng.integers(0, 40, 64),
                          "label": rng.integers(0, 2, 64)})
    train_ce(pairs, texts, d / "m", bs=16, model_name=str(d / "init"))
    p = score_ce(pairs.select("s1_idx", "c_idx"), texts, d / "m", bs=20)
    assert p.shape == (64,) and p.dtype == np.float32 and ((p >= 0) & (p <= 1)).all()
    print("selftest OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
        sys.exit()
    from ber.run import WORK
    cmd, split = a[0], (a[a.index("--split") + 1] if "--split" in a else "train")
    prep = pl.read_parquet(WORK / f"{'train' if cmd == 'train' else split}_prep.parquet")
    texts = build_text(prep)
    if cmd == "train":
        train_ce(pl.read_parquet(WORK / "ce_fit_pairs.parquet"), texts, WORK / "ce_model",
                 bs=int(os.environ.get("CE_BS", 64)), lr=float(os.environ.get("CE_LR", 1e-5)),
                 max_pairs=int(os.environ.get("CE_MAX", 6_000_000)))
    else:
        pr = pl.read_parquet(WORK / f"ce_{split}_pairs.parquet").select("s1_idx", "c_idx")
        pr.with_columns(pl.Series("ce", score_ce(pr, texts, WORK / "ce_model", bs=int(os.environ.get("CE_BS", 64)) * 4), dtype=pl.Float32)) \
          .write_parquet(WORK / f"ce_{split}_scores.parquet")
        log("wrote", split, pr.height)
