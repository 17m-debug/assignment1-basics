"""
tokenizer 作业自检脚本。

用法:
    uv run python check_tokenizer.py            # 全部检查
    uv run python check_tokenizer.py --quick    # 跳过耗时项（吞吐量 / 内存）

检查内容:
  A. 官方 pytest（test_tokenizer.py + test_train_bpe.py）
  B. 边界用例：空串 / 无 special / 首尾与相邻 special / 重叠 special / CRLF / emoji
  C. 健壮性：special 不在 vocab、不修改传入的 vocab、from_files 往返
  D. encode_iterable 与 encode 一致，且是真流式（用 tracemalloc 测峰值）
  E. 真实语料：往返一致、压缩率、吞吐量
"""

import pickle
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from cs336_basics.tokenizer import tokenizer as Tokenizer

ROOT = Path(__file__).parent
OUT_TS = ROOT / "bpe_tinystories_out"
DATA_TS = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
EOT = "<|endoftext|>"
QUICK = "--quick" in sys.argv

results = []  # (name, ok, detail)


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -> {detail}" if (detail and not cond) else ""))


def section(title):
    print(f"\n=== {title} ===")


def load_ts(special=(EOT,)):
    vocab = pickle.load(open(OUT_TS / "vocab.pkl", "rb"))
    merges = pickle.load(open(OUT_TS / "merges.pkl", "rb"))
    return vocab, merges, Tokenizer(vocab, merges, list(special))


# ---------------------------------------------------------------- A
def run_pytest():
    section("A. 官方 pytest")
    for f in ("tests/test_tokenizer.py", "tests/test_train_bpe.py"):
        p = subprocess.run(
            ["uv", "run", "pytest", f, "-q", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True,
        )
        tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-1:]
        check(f"pytest {f}", p.returncode == 0, " | ".join(tail) or p.stderr[-300:])
        print("        ", tail[0] if tail else "")


# ---------------------------------------------------------------- B
def edge_cases():
    section("B. 边界用例")
    try:
        vocab, merges, t = load_ts()
    except FileNotFoundError:
        check("加载 bpe_tinystories_out/*.pkl", False, "文件不存在，请先跑 run_tinystories_bpe.py")
        return

    def rt(name, s, tok=t):
        try:
            ids = tok.encode(s)
            check(name, tok.decode(ids) == s, f"decode(encode(s)) != s; s={s!r}")
            return ids
        except Exception as e:  # noqa
            check(name, False, f"{type(e).__name__}: {e}")

    rt("空字符串往返", "")
    check("空字符串编码为 []", t.encode("") == [])
    rt("单字符", "a")
    rt("纯空白", "   \n\t  ")
    rt("CRLF 换行", "line1\r\nline2\r\n")
    rt("emoji / 多字节", "héllo 世界 🙂 ñ")
    rt("连续空格与缩进", "a    b\n\n\n    c")
    rt("缩写 's 're n't", "I'm sure they're here, isn't it? We've gone.")
    rt("长重复串", "-" * 500 + "a" * 300)

    eot_id = t.id_map[EOT.encode()]
    ids = rt("special 在中间", f"hi{EOT}bye")
    check("中间的 special 恰为单个 id", ids is not None and ids.count(eot_id) == 1)
    ids = rt("special 在开头", f"{EOT}hi")
    check("开头 special 为首个 id", ids is not None and ids[0] == eot_id)
    ids = rt("special 在结尾", f"hi{EOT}")
    check("结尾 special 为末个 id", ids is not None and ids[-1] == eot_id)
    ids = rt("相邻 special", f"{EOT}{EOT}{EOT}")
    check("三个相邻 special -> 三个 id", ids == [eot_id] * 3, f"得到 {ids}")
    ids = rt("special 两侧有换行", f"a\n{EOT}\nb")
    check("换行不吞 special", ids is not None and ids.count(eot_id) == 1)

    # 无 special
    t0 = Tokenizer(vocab, merges, None)
    ids = rt("无 special：普通往返", "Hello world.", t0)
    ids = t0.encode(EOT)
    check("无 special 时 <|endoftext|> 被拆成多个 token", len(ids) > 1, f"得到 {ids}")
    t0b = Tokenizer(vocab, merges, [])
    check("special=[] 与 None 等价", t0b.encode("a b") == t0.encode("a b"))

    # 重叠 special
    ov_vocab = dict(vocab)
    a, b = "<|a|>", "<|a|><|a|>"
    ov_vocab[len(ov_vocab)] = a.encode()
    ov_vocab[len(ov_vocab)] = b.encode()
    tv = Tokenizer(ov_vocab, merges, [a, b])
    ids = tv.encode(f"x{b}y")
    check("重叠 special：长的优先", ids.count(tv.id_map[b.encode()]) == 1 and tv.id_map[a.encode()] not in ids,
          f"ids={ids}")
    check("重叠 special：往返", tv.decode(ids) == f"x{b}y")
    # 与传入顺序无关
    tv2 = Tokenizer(ov_vocab, merges, [b, a])
    check("special 传入顺序无关", tv2.encode(f"x{b}y") == ids)


# ---------------------------------------------------------------- C
def robustness():
    section("C. 健壮性")
    vocab, merges, _ = load_ts()

    # 不修改传入 vocab
    v_copy = dict(vocab)
    Tokenizer(vocab, merges, [EOT, "<|extra|>"])
    check("构造函数不修改传入的 vocab", vocab == v_copy,
          f"vocab 长度 {len(v_copy)} -> {len(vocab)}")

    # special 不在 vocab
    try:
        t = Tokenizer(dict(vocab), merges, ["<|brand_new|>"])
        ids = t.encode("a<|brand_new|>b")
        check("special 不在 vocab 时能编码", len(ids) > 0 and t.decode(ids) == "a<|brand_new|>b")
        n_new = sum(1 for x in ids if t.vocab.get(x) == b"<|brand_new|>")
        check("该 special 被编成单个 token", n_new == 1, f"出现次数 {n_new}")
    except KeyError as e:
        check("special 不在 vocab 时能编码", False,
              f"KeyError {e}。__init__ 需要在 special 缺失时补充进 vocab（复制一份，别改传入的字典）")
    except Exception as e:  # noqa
        check("special 不在 vocab 时能编码", False, f"{type(e).__name__}: {e}")

    # from_files
    try:
        tf = Tokenizer.from_files(OUT_TS / "vocab.pkl", OUT_TS / "merges.pkl", [EOT])
        _, _, tm = load_ts()
        s = f"Once upon a time.{EOT}The end."
        check("from_files 与直接构造结果一致", tf.encode(s) == tm.encode(s))
    except Exception as e:  # noqa
        check("from_files 可用", False, f"{type(e).__name__}: {e}")

    # 多次调用结果一致（缓存不污染）
    _, _, t = load_ts()
    s = "the cat sat on the mat. the cat sat."
    check("重复编码结果一致（缓存正确）", t.encode(s) == t.encode(s) == t.encode(s))

    # decode 非法 UTF-8 不抛异常
    try:
        out = t.decode([t.id_map[b"\xe4"]])
        check("decode 对残缺 UTF-8 不抛异常", isinstance(out, str))
    except Exception as e:  # noqa
        check("decode 对残缺 UTF-8 不抛异常", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- D
def iterable_checks():
    section("D. encode_iterable")
    if not DATA_TS.exists():
        check("存在 TinyStories valid 数据", False, str(DATA_TS))
        return
    _, _, t = load_ts()

    with open(DATA_TS, encoding="utf-8") as f:
        head = "".join(f.readline() for _ in range(3000))

    lines = head.splitlines(keepends=True)
    streamed = list(t.encode_iterable(iter(lines)))
    per_line = [x for l in lines for x in t.encode(l)]
    check("encode_iterable == 逐项 encode 后拼接", streamed == per_line,
          f"长度 {len(streamed)} vs {len(per_line)}")

    # 以文档为单位切块（在 <|endoftext|> 之后切），应与整段 encode 完全一致
    import regex as _re
    full = t.encode(head)
    docs = [d for d in _re.split(r"(?<=<\|endoftext\|>)", head) if d]
    via_docs = list(t.encode_iterable(iter(docs)))
    check("encode_iterable(按文档切块) == encode(整段)", via_docs == full,
          f"长度 {len(via_docs)} vs {len(full)}")
    print("        (按行喂入时，跨行的空白如 '\\n'+'\\n' 无法合并成 '\\n\\n'，"
          "与整段编码会有少量差异，属预期；大语料请按文档切块)")
    check("返回的是迭代器/生成器（惰性）",
          hasattr(t.encode_iterable(iter(["a"])), "__next__"))

    if QUICK:
        print("  (--quick: 跳过内存测试)")
        return

    # 流式内存：全量文件逐行，峰值应远小于文件大小
    size_mb = DATA_TS.stat().st_size / 1e6
    _, _, t2 = load_ts()
    tracemalloc.start()
    n = 0
    with open(DATA_TS, encoding="utf-8") as f:
        for _ in t2.encode_iterable(f):
            n += 1
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1e6
    check(f"流式编码 {size_mb:.0f}MB 文件，峰值内存 {peak_mb:.1f}MB < 文件大小的 1/2",
          peak_mb < size_mb / 2, f"峰值 {peak_mb:.1f}MB")
    cache_n = len(t2.cache)
    print(f"        缓存条目数 {cache_n:,}（无上限增长；大语料上可能成为内存隐患）")
    check("缓存条目数在合理范围 (<200 万)", cache_n < 2_000_000, f"{cache_n:,}")


# ---------------------------------------------------------------- E
def real_corpus():
    section("E. 真实语料")
    if not DATA_TS.exists():
        check("存在 TinyStories valid 数据", False, str(DATA_TS))
        return
    _, _, t = load_ts()
    with open(DATA_TS, encoding="utf-8") as f:
        text = f.read(2_000_000)  # 前 2MB
    # 在文档边界截断，避免截在 special token 中间
    cut = text.rfind(EOT)
    text = text[: cut + len(EOT)] if cut > 0 else text

    t0 = time.time()
    ids = t.encode(text)
    dt = time.time() - t0
    nbytes = len(text.encode("utf-8"))
    check("2MB 样本往返一致", t.decode(ids) == text)
    ratio = nbytes / len(ids)
    print(f"        压缩率 {ratio:.2f} bytes/token（作业参考：TinyStories 约 4 左右）")
    check("压缩率在 3~6 之间", 3 <= ratio <= 6, f"{ratio:.2f}")
    thr = nbytes / dt / 1e6
    print(f"        吞吐量 {thr:.2f} MB/s（整份 TinyStories train 约 2GB）")
    if not QUICK:
        eta_min = 2200 / thr / 60 if thr > 0 else float("inf")
        print(f"        按此速度编码 ~2.2GB 训练集约需 {eta_min:.0f} 分钟（单进程）")

    # 所有 id 合法
    check("所有 id 都在 vocab 范围内", all(0 <= i < len(t.vocab) for i in ids[:200000]))

    # 与 tiktoken 对照（仅当 vocab 为 gpt2 时有意义，这里只做可用性提示）
    try:
        import tiktoken  # noqa
        print("        tiktoken 已安装（官方 pytest 已与其对照）")
    except Exception:
        pass


def main():
    run_pytest()
    edge_cases()
    robustness()
    iterable_checks()
    real_corpus()

    section("汇总")
    bad = [r for r in results if not r[1]]
    print(f"  通过 {len(results) - len(bad)} / {len(results)}")
    if bad:
        print("\n  未通过项:")
        for name, _, detail in bad:
            print(f"   - {name}" + (f"\n       {detail}" if detail else ""))
        sys.exit(1)
    print("  全部通过")


if __name__ == "__main__":
    main()
