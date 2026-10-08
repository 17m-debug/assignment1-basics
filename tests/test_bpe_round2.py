"""
第 ② 级验证：手工构造 counts，只测 merge 算法本身。
不读文件、不跑正则 —— 出 bug 能立刻定位到算法。

运行：python test_bpe_round2.py
"""

# ---------- 1. 测试数据（讲义给的 4 词例子，空格分割、空格丢弃）----------
counts = {
    (b'l', b'o', b'w'): 5,
    (b'l', b'o', b'w', b'e', b'r'): 2,
    (b'w', b'i', b'd', b'e', b's', b't'): 3,
    (b'n', b'e', b'w', b'e', b's', b't'): 6,
}

NUM_MERGES = 6
EXPECTED = ['st', 'est', 'ow', 'low', 'west', 'ne']   # 讲义给的标准答案


# ---------- 2. 在这里粘贴你的 merge_in_tuple（记得补 i = 0）----------
# def merge_in_tuple(pt, a, b):
#     out = []
#     i = 0                      # ← 你漏了这个
#     while i < len(pt):
#         if i < len(pt) - 1 and pt[i] == a and pt[i + 1] == b:
#             out.append(a + b)
#             i += 2
#         else:
#             out.append(pt[i])
#             i += 1
#     return tuple(out)


# ---------- 3. 在这里写你的 merge 循环 ----------
# 每轮要做的事：
#   pairs = {}                                    每轮必须清空
#   for pt, cnt in counts.items():                ← 注意 .items()
#       for i in range(len(pt) - 1):
#           pairs[(pt[i], pt[i+1])] += cnt        ← 加的是 cnt 不是 1
#   if not pairs: break                           ← 终止条件
#   max_pair = max(pairs, key=lambda p: (pairs[p], p))     ← tie-break
#   merges.append(max_pair)                       ← 记录
#   new_counts = {}                               重建字典，避免遍历中修改
#   for pt, cnt in counts.items():
#       new_counts[merge_in_tuple(pt, *max_pair)] += cnt
#   counts = new_counts

merges = []        # ← 你必须在这里记录


# ---------- 4. 自动检查 ----------
if __name__ == "__main__":
    if not merges:
        print("还没写循环：merges 是空的。去第 3 步把循环补上。")
    else:
        got = [(a + b).decode("utf-8") for a, b in merges]
        print("你得到的 merge 产物:", got)
        print("期望的答案      :", EXPECTED)
        print()
        if got == EXPECTED:
            print("✅ 通过 —— merge 算法正确，可以接全流程了。")
        else:
            print("❌ 不一致。逐轮对比：")
            for i, (g, e) in enumerate(zip(got, EXPECTED)):
                flag = "OK" if g == e else f"← 期望 {e}"
                print(f"   第{i+1}轮: 得到 {g!r:8} {flag}")
            if len(got) != len(EXPECTED):
                print(f"   长度也不同: {len(got)} vs {len(EXPECTED)}")

        # 看看每轮的 tie-break 有没有踩坑
        print()
        print("merges 原始结构（每项是两个部分，不是产物）:")
        for m in merges:
            print("   ", m, "-> 产物", (m[0] + m[1]))
