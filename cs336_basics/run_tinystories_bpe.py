"""
在 TinyStories 上训练 byte-level BPE（vocab_size=10000），并输出题目要求的统计量。

用法（在项目根目录跑）：
    uv run python run_tinystories_bpe.py
    uv run python run_tinystories_bpe.py data/TinyStoriesV2-GPT4-valid.txt
    uv run python run_tinystories_bpe.py data/TinyStoriesV2-GPT4-train.txt 10000

输出：
    1. 训练耗时
    2. 峰值内存（RSS）
    3. 最长 token 及其长度
    4. vocab / merges 序列化到磁盘

说明：本脚本只做「调用 + 测量 + 落盘」，
     BPE 实现仍在你自己的 cs336_basics/train_bpe.py 里。
"""

import json
import pickle
import sys
import threading
import time
from pathlib import Path

import psutil

# ---------- 导入你的实现 ----------
try:
    from cs336_basics.train_bpe import train_bpe
except ImportError as e:
    print("❌ 无法导入 cs336_basics.train_bpe：", e)
    print("   请在项目根目录运行：uv run python run_tinystories_bpe.py")
    sys.exit(1)


# ---------- 参数 ----------
DEFAULT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATH)
vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
special_tokens = ["<|endoftext|>"]

if not input_path.exists():
    print(f"❌ 找不到数据文件: {input_path}")
    print("   先在 hf-mirror.com 下载 TinyStories，或用 valid 集做调试。")
    sys.exit(1)

size_mb = input_path.stat().st_size / 1024 / 1024
print(f"输入: {input_path}  ({size_mb:.1f} MB)")
print(f"vocab_size: {vocab_size}  special_tokens: {special_tokens}")
print()


# ---------- 内存采样（后台线程）----------
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
print("开始训练 ...", flush=True)
start = time.time()
vocab, merges = train_bpe(
    input_path=str(input_path),
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)
elapsed = time.time() - start
_stop.set()
t.join(timeout=1)


# ---------- 统计量 ----------
peak_gb = peak_rss / 1024**3
print()
print("=" * 60)
print(f"训练耗时     : {elapsed:.1f} 秒  ({elapsed/60:.2f} 分钟)")
print(f"峰值内存 RSS : {peak_gb:.2f} GB")
print(f"vocab 大小   : {len(vocab)}")
print(f"merges 数量  : {len(merges)}")
print("=" * 60)

# 最长 token
longest = max(vocab.values(), key=len)
print()
print(f"最长 token 长度 : {len(longest)} 字节")
print(f"最长 token 内容 : {longest!r}")
print(f"可读形式        : {longest.decode('utf-8', errors='replace')}")

# 前 10 长的 token（看规律）
print()
print("最长的 10 个 token：")
for b in sorted(vocab.values(), key=len, reverse=True)[:10]:
    print(f"   {len(b):3d} 字节  {b.decode('utf-8', errors='replace')!r}")

# 长度分布
from collections import Counter
dist = Counter(len(b) for b in vocab.values())
print()
print("token 长度分布（长度: 个数）：")
for L in sorted(dist)[:16]:
    print(f"   {L:3d}: {dist[L]}")


# ---------- 序列化 ----------
out_dir = Path("bpe_tinystories_out")
out_dir.mkdir(exist_ok=True)

# vocab: JSON（bytes 用 latin-1 无损转 str，可 json 化且能原样还原）
vocab_json = {str(i): b.decode("latin-1") for i, b in vocab.items()}
(out_dir / "vocab.json").write_text(
    json.dumps(vocab_json, ensure_ascii=False), encoding="utf-8"
)

# merges: 每行 "tok1 tok2"，同样用 latin-1
with (out_dir / "merges.txt").open("w", encoding="utf-8") as f:
    for a, b in merges:
        f.write(f"{a.decode('latin-1')} {b.decode('latin-1')}\n")

# 同时存一份 pickle，读回来就是原始 bytes 类型
with (out_dir / "vocab.pkl").open("wb") as f:
    pickle.dump(vocab, f)
with (out_dir / "merges.pkl").open("wb") as f:
    pickle.dump(merges, f)

print()
print(f"已序列化到 {out_dir}/  -> vocab.json, merges.txt, vocab.pkl, merges.pkl")


# ---------- 题目要求的 1-2 句回答模板 ----------
print()
print("-" * 60)
print("")
print()
print(
    f"Training took {elapsed/60:.1f} minutes and peaked at {peak_gb:.1f} GB of RAM; "
    f"the longest token in the vocabulary is {len(longest)} bytes "
    f"({longest.decode('utf-8', errors='replace')!r}), "
    "which [does/does not] make sense because ..."
)
print("-" * 60)
