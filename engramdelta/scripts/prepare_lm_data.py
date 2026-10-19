"""Phase 2 data preparation (NB03, CPU only: uses no GPU quota).

    python -m engramdelta.scripts.prepare_lm_data --out /kaggle/working/lmdata \
        --train-tokens 150000000 --val-docs 2000 --vocab 16384

Steps:
  1. Stream FineWeb-Edu (HuggingFaceFW/fineweb-edu, config sample-10BT) in its fixed order.
  2. The first --val-docs documents form the validation set. Their text hashes are kept, and any later
     document with the same text is dropped, so val never overlaps train.
  3. Train a byte-level BPE tokenizer (--vocab, one special token <|endoftext|>) on the next --tok-docs TRAIN
     documents only.
  4. Tokenize val, then train documents until --train-tokens. Each document ends with <|endoftext|>.
     Written as uint16 memmaps (val.bin, train.bin).
  5. Build the Engram tokenizer-compression lookup (port of the reference CompressedTokenizer).
  6. Write meta.json with counts, the source and the dedup statistics.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from ..models.engram import build_compressed_lookup

EOS = "<|endoftext|>"


def doc_stream(dataset: str, config: str):
    from datasets import load_dataset
    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    for row in ds:
        text = row.get("text") or ""
        if text.strip():
            yield text


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def train_tokenizer(texts, vocab: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOS], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def write_tokens(tok, texts, path: str, eos_id: int, max_tokens: int = None, batch: int = 512) -> int:
    n = 0
    with open(path, "wb") as f:
        buf = []
        def flush(buf):
            nonlocal n
            for enc in tok.encode_batch(buf):
                ids = enc.ids + [eos_id]
                if max_tokens is not None and n + len(ids) > max_tokens:
                    ids = ids[:max_tokens - n]
                np.asarray(ids, dtype=np.uint16).tofile(f)
                n += len(ids)
                if max_tokens is not None and n >= max_tokens:
                    return True
            return False
        for t in texts:
            buf.append(t)
            if len(buf) == batch:
                if flush(buf):
                    return n
                buf = []
        if buf:
            flush(buf)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--train-tokens", type=int, default=150_000_000)
    ap.add_argument("--val-docs", type=int, default=2000)
    ap.add_argument("--tok-docs", type=int, default=50_000)
    ap.add_argument("--vocab", type=int, default=16384)
    args = ap.parse_args(argv)
    assert args.vocab <= 65535, "uint16 storage"
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    stream = doc_stream(args.dataset, args.config)

    val_docs = [next(stream) for _ in range(args.val_docs)]
    val_hashes = {sha(t) for t in val_docs}
    stats = {"dropped_duplicates_of_val": 0}

    def train_docs():
        for t in stream:
            if sha(t) in val_hashes:
                stats["dropped_duplicates_of_val"] += 1
                continue
            yield t

    train_iter = train_docs()
    tok_docs = [next(train_iter) for _ in range(args.tok_docs)]
    tok = train_tokenizer(tok_docs, args.vocab)
    eos_id = tok.token_to_id(EOS)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"tokenizer trained on {len(tok_docs)} train docs ({time.time() - t0:.0f}s), vocab {tok.get_vocab_size()}")

    n_val = write_tokens(tok, val_docs, os.path.join(args.out, "val.bin"), eos_id)
    print(f"val: {len(val_docs)} docs, {n_val:,} tokens")

    def all_train():
        yield from tok_docs          # the tokenizer docs are train docs too
        yield from train_iter
    n_train = write_tokens(tok, all_train(), os.path.join(args.out, "train.bin"), eos_id, args.train_tokens)
    print(f"train: {n_train:,} tokens ({time.time() - t0:.0f}s)")

    lookup, n_canonical = build_compressed_lookup(tok)
    np.save(os.path.join(args.out, "engram_compress.npy"), lookup)
    meta = {"vocab_size": tok.get_vocab_size(), "eos_id": eos_id, "train_tokens": n_train, "val_tokens": n_val,
            "val_docs": len(val_docs), "tokenizer_docs": len(tok_docs), "dataset": args.dataset,
            "dataset_config": args.config, "engram_canonical_ids": n_canonical, **stats,
            "split_rule": "first val_docs documents of the stream = val; later documents with identical text dropped"}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
