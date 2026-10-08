"""
定位 BPE 训练的真实瓶颈（不要再靠猜）。

用法:
    uv run python profile_bpe.py                       # 默认 owt_sample.txt, vocab 32000
    uv run python profile_bpe.py data/owt_sample.txt 8000

做什么:
  1. 跑一次带计时的训练，报告三段时间
  2. 报告规模：互异 pre-token 数、pair 种类数
  3. 用 cProfile 列出最耗时的函数（前 25 行）
"""

import cProfile
import pstats
import sys
import time
import io
from collections import Counter
from pathlib import Path

import regex as re

# ---------- 你的实现 ----------
from cs336_basics.train_bpe import merge_in_tuple, train_bpe  # noqa

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def build_tables(counts):
    """复刻你 train_bpe 里的构建阶段，顺便量规模和时间"""
    t0 = time.time()
    pretokens = list(counts.keys())
    freqs = list(counts.values())
    pair_counts = {}
    pair_to_ids = {}
    for i in range(len(pretokens)):
        f = freqs[i]
        pt = pretokens[i]
        for j in range(1, len(pt)):
            p = (pt[j - 1], pt[j])
            pair_counts[p] = pair_counts.get(p, 0) + f
            pair_to_ids.setdefault(p, set()).add(i)
    t1 = time.time()
    return pretokens, freqs, pair_counts, pair_to_ids, t1 - t0


def scan_counts(path, special_tokens):
    """复刻 pre-tokenization，量时间"""
    t0 = time.time()
    data = Path(path).read_text(encoding="utf-8")
    pat_special = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
    chunks = re.split(pat_special, data)
    counts = {}
    for chunk in chunks:
        if not chunk:
            continue
        for m in re.finditer(PATTERN, chunk):
            pt = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            counts[pt] = counts.get(pt, 0) + 1
    t1 = time.time()
    return counts, t1 - t0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/owt_sample.txt"
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32000
    special_tokens = ["<|endoftext|>"]

    if not Path(path).exists():
        print(f"ERROR: {path} not found")
        return

    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"Input: {path}  ({size_mb:.1f} MB)   vocab_size={vocab_size}")
    print()

    # ---- 1. pre-tokenization ----
    print("[1] Scanning / pre-tokenizing ...", flush=True)
    counts, t_scan = scan_counts(path, special_tokens)
    print(f"    pre-tokenize time : {t_scan:.1f} s")
    print(f"    distinct pre-tokens: {len(counts):,}")
    print()

    # ---- 2. build ----
    print("[2] Building tables ...", flush=True)
    pretokens, freqs, pair_counts, pair_to_ids, t_build = build_tables(counts)
    print(f"    build time   : {t_build:.1f} s")
    print(f"    pair types   : {len(pair_counts):,}")
    print(f"    avg set size : {sum(len(s) for s in pair_to_ids.values()) / max(1, len(pair_to_ids)):.1f}")
    print()

    # ---- 3. 只测 max() 这一步的代价 ----
    print("[3] Measuring cost of ONE max() over pair_counts ...", flush=True)
    t0 = time.time()
    for _ in range(5):
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
    t_max = (time.time() - t0) / 5
    num_merges = vocab_size - 256 - len(special_tokens)
    print(f"    one max()        : {t_max*1000:.1f} ms")
    print(f"    x {num_merges} rounds: {t_max*num_merges/60:.1f} min  <-- 光选最大就要这么久")
    print()

    # ---- 4. cProfile ----
    print("[4] cProfile on full train_bpe ...", flush=True)
    pr = cProfile.Profile()
    pr.enable()
    train_bpe(input_path=str(path), vocab_size=vocab_size, special_tokens=special_tokens)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main()
