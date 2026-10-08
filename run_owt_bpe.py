"""
在 OpenWebText 上训练 byte-level BPE（vocab_size=32000），并与 TinyStories 结果对比。

用法（项目根目录）：
    uv run python run_owt_bpe.py
    uv run python run_owt_bpe.py data/owt_train.txt
    uv run python run_owt_bpe.py data/owt_valid.txt 32000

输出：训练耗时、峰值内存、最长 token、长度分布、序列化；
     若已跑过 TinyStories，自动做对比。
"""

import json
import pickle
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import psutil

try:
    from cs336_basics.train_bpe import train_bpe
except ImportError as e:
    print("ERROR: cannot import cs336_basics.train_bpe:", e)
    print("       Run from project root: uv run python run_owt_bpe.py")
    sys.exit(1)


# ---------- 参数 ----------
DEFAULT_PATH = "data/owt_train.txt"
input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATH)
vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32_000
special_tokens = ["<|endoftext|>"]

if not input_path.exists():
    print(f"ERROR: data file not found: {input_path}")
    sys.exit(1)

size_mb = input_path.stat().st_size / 1024 / 1024
print(f"Input : {input_path}  ({size_mb:.1f} MB)")
print(f"vocab : {vocab_size}   special: {special_tokens}")
print()


# ---------- 内存采样 ----------
proc = psutil.Process()
peak_rss = 0
_stop = threading.Event()


def _sample():
    global peak_rss
    while not _stop.is_set():
        try:
            peak_rss = max(peak_rss, proc.memory_info().rss)
        except Exception:
            pass
        time.sleep(0.05)


t = threading.Thread(target=_sample, daemon=True)
t.start()


# ---------- 训练 ----------
print("Training ...", flush=True)
start = time.time()
vocab, merges = train_bpe(
    input_path=str(input_path),
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)
elapsed = time.time() - start
_stop.set()
t.join(timeout=1)


# ---------- 统计 ----------
peak_gb = peak_rss / 1024**3
print()
print("=" * 60)
print(f"Time        : {elapsed:.1f} s  ({elapsed/60:.2f} min)")
print(f"Peak RSS    : {peak_gb:.2f} GB")
print(f"vocab size  : {len(vocab)}")
print(f"merges      : {len(merges)}")
print("=" * 60)

longest = max(vocab.values(), key=len)
print()
print(f"Longest token : {len(longest)} bytes")
print(f"Content       : {longest!r}")
print(f"Readable      : {longest.decode('utf-8', errors='replace')!r}")

print()
print("Top-10 longest tokens:")
for b in sorted(vocab.values(), key=len, reverse=True)[:10]:
    print(f"   {len(b):3d}  {b.decode('utf-8', errors='replace')!r}")

dist = Counter(len(b) for b in vocab.values())
print()
print("Length distribution (len: count):")
for L in sorted(dist)[:20]:
    print(f"   {L:3d}: {dist[L]}")


# ---------- 序列化 ----------
out_dir = Path("bpe_owt_out")
out_dir.mkdir(exist_ok=True)

vocab_json = {str(i): b.decode("latin-1") for i, b in vocab.items()}
(out_dir / "vocab.json").write_text(
    json.dumps(vocab_json, ensure_ascii=False), encoding="utf-8"
)
with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
    for a, b in merges:
        f.write(f"{a.decode('latin-1')} {b.decode('latin-1')}\n")
with (out_dir / "vocab.pkl").open("wb") as f:
    pickle.dump(vocab, f)
with (out_dir / "merges.pkl").open("wb") as f:
    pickle.dump(merges, f)

print()
print(f"Serialized to {out_dir}/  -> vocab.json, merges.txt, vocab.pkl, merges.pkl")


# ---------- (b) 与 TinyStories 对比 ----------
print()
print("-" * 60)
print("COMPARISON with TinyStories tokenizer (if available)")
print("-" * 60)

ts_dir = Path("bpe_tinystories_out")
if not (ts_dir / "vocab.pkl").exists():
    print("TinyStories output not found. Run run_tinystories_bpe.py first.")
else:
    with (ts_dir / "vocab.pkl").open("rb") as f:
        ts_vocab = pickle.load(f)
    with (ts_dir / "merges.pkl").open("rb") as f:
        ts_merges = pickle.load(f)

    ts_set = set(ts_vocab.values())
    owt_set = set(vocab.values())
    shared = ts_set & owt_set

    print(f"TinyStories vocab : {len(ts_set)}")
    print(f"OpenWebText vocab : {len(owt_set)}")
    print(f"Shared tokens     : {len(shared)}")

    # 只在 OWT 里出现的（新闻/网络语料特有）
    only_owt = sorted(owt_set - ts_set, key=len, reverse=True)[:15]
    print()
    print("Tokens ONLY in OpenWebText (web/news flavour):")
    for b in only_owt:
        print(f"   {b.decode('utf-8', errors='replace')!r}")

    only_ts = sorted(ts_set - owt_set, key=len, reverse=True)[:15]
    print()
    print("Tokens ONLY in TinyStories (children's story flavour):")
    for b in only_ts:
        print(f"   {b.decode('utf-8', errors='replace')!r}")

    # 平均 token 长度（压缩率代理指标）
    ts_avg = sum(len(b) for b in ts_set) / len(ts_set)
    owt_avg = sum(len(b) for b in owt_set) / len(owt_set)
    print()
    print(f"Avg token length  TinyStories: {ts_avg:.2f} bytes")
    print(f"Avg token length  OpenWebText: {owt_avg:.2f} bytes")

print()
print("=" * 60)
print("Answer template for (a):")
print(
    f"Training took {elapsed/60:.1f} minutes and peaked at {peak_gb:.1f} GB; "
    f"the longest token is {len(longest)} bytes "
    f"({longest.decode('utf-8', errors='replace')!r}), which ... "
)
print("Answer template for (b):")
print("Compared to TinyStories, the OpenWebText tokenizer spends its capacity on ...")
print("=" * 60)
