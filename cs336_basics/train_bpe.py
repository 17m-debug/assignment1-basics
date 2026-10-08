import regex as re
import time
from collections import Counter


def merge_in_tuple(pt, a, b):
    i = 0
    out = []
    while i < len(pt):
        if i < len(pt) - 1 and pt[i] == a and pt[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(pt[i])
            i += 1
    return tuple(out)

def run_merges(pair_counts, pair_to_ids, pretokens, freqs, num_merges):
    merges = []
    for _ in range(num_merges):
        if not pair_counts: break
        max_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(max_pair)
        affected = list(pair_to_ids[max_pair])
        for pid in affected:
            for p, k in Counter(zip(pretokens[pid], pretokens[pid][1:])).items():
                pair_counts[p] -= k * freqs[pid]
                if pair_counts[p] == 0:
                    del pair_counts[p]
                pair_to_ids[p].discard(pid)
            new_pt = merge_in_tuple(pretokens[pid], max_pair[0], max_pair[1])
            pretokens[pid] = new_pt
            for p, k in Counter(zip(new_pt, new_pt[1:])).items():
                pair_counts[p] = pair_counts.get(p, 0) + k * freqs[pid]
                pair_to_ids.setdefault(p, set()).add(pid)
    return merges
    

def run_vocab(special_tokens, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    for a, b in merges:
        vocab[next_id] = a + b
        next_id += 1
    return vocab


def train_bpe(input_path, vocab_size, special_tokens):
    num_merges = vocab_size - 256 - len(special_tokens)
    # input sth and read the file and spilt
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    pat_special = "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True)))
    pattern =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    chunks = re.split(pat_special, data)
    t0 = time.time()
    counts = {}
    for chunk in chunks:
        if not chunk: continue
        for m in re.finditer(pattern, chunk):
            pt = tuple(bytes([b]) for b in m.group().encode("utf-8"))
            counts[pt] = counts.get(pt, 0) + 1
    t1 = time.time()

    pretokens = list(counts.keys())
    freqs = list(counts.values())
    pair_counts = {}
    pair_to_ids = {}
    for i in range(len(freqs)):
        for j in range(1, len(pretokens[i])):
            p = (pretokens[i][j-1], pretokens[i][j])
            pair_counts[p] = pair_counts.get(p, 0) + freqs[i]
            pair_to_ids.setdefault(p, set()).add(i)
    merges = run_merges(pair_counts, pair_to_ids, pretokens, freqs, num_merges)

    t2 = time.time()
    print(f"Time taken for merges: {t2 - t1}")
    # remerge for enough times
    vocab = run_vocab(special_tokens, merges)
    # merge and get subtoken
    return vocab, merges