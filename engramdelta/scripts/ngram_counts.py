"""Training-set frequency of the n-gram Engram looks up at every val token (study E2.ngram, PREREG.md).

    python -m engramdelta.scripts.ngram_counts --data results/lmdata

At val input position j, Engram's order-n head hashes the canonical ids (engram_compress.npy) of the n tokens
ending at j: (c[j-n+1], ..., c[j]). This script counts how often each such n-gram (n = 2, 3) occurs in train.bin
(canonical ids, all positions, across document boundaries as Engram itself does) and writes
<data>/val_ngram_counts.npz with uint32 arrays `bigram` and `trigram` of length len(val.bin): the train count of
the n-gram ending at that val token (0 for the first n-1 tokens of val.bin). Eval only uses positions p >= 2 inside
a window, so the n-gram never crosses a window start.
"""
import argparse
import os
import sys
import time

import numpy as np

OUT_NAME = "val_ngram_counts.npz"


def ngram_keys(c: np.ndarray, n: int, base: int) -> np.ndarray:
    """int64 key of the n-gram ending at each position i >= n-1 (length len(c) - n + 1)."""
    k = c[:len(c) - n + 1].astype(np.int64)
    for s in range(1, n):
        k = k * base + c[s:len(c) - n + 1 + s]
    return k


def counts_for(train_c: np.ndarray, val_c: np.ndarray, n: int, base: int) -> np.ndarray:
    uniq, cnt = np.unique(ngram_keys(train_c, n, base), return_counts=True)
    q = ngram_keys(val_c, n, base)
    i = np.searchsorted(uniq, q).clip(max=len(uniq) - 1)
    hit = uniq[i] == q
    out = np.zeros(len(val_c), dtype=np.uint32)
    out[n - 1:] = np.where(hit, cnt[i], 0).astype(np.uint32)
    return out


def build(data: str) -> str:
    compress = np.load(os.path.join(data, "engram_compress.npy")).astype(np.int64)
    base = int(compress.max()) + 1
    assert base ** 3 < 2 ** 62
    train_c = compress[np.memmap(os.path.join(data, "train.bin"), dtype=np.uint16, mode="r")]
    val_c = compress[np.memmap(os.path.join(data, "val.bin"), dtype=np.uint16, mode="r")]
    res = {}
    for name, n in (("bigram", 2), ("trigram", 3)):
        t0 = time.time()
        res[name] = counts_for(train_c, val_c, n, base)
        print(f"[ngram_counts] {name}: {time.time() - t0:.0f}s | val tokens with count 0: "
              f"{(res[name][n - 1:] == 0).mean():.3f} | median count {int(np.median(res[name][n - 1:]))}", flush=True)
    out = os.path.join(data, OUT_NAME)
    np.savez(out, **res)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="LM data folder (notebook 03 output)")
    print("[ngram_counts] wrote", build(ap.parse_args(argv).data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
