#!/usr/bin/env python3
"""
Compute inter-annotator agreement for span-tagged text like:

    Novelist: <REASONING>I've been given the book</REASONING> <CORRECT_INFORMATION>...

Assumes both annotators tagged (a version of) the SAME underlying text with
non-nested <LABEL>...</LABEL> spans, and that untagged text is implicitly "O".

Outputs:
  - token-level agreement: raw accuracy, Cohen's Kappa, per-label P/R/F1
  - span-level agreement: exact-match P/R/F1 per label (annotator1 vs annotator2)
"""

import re
import sys
import argparse
from collections import defaultdict, Counter
import os
import pandas as pd
from pprint import pp

TAG_RE = re.compile(r"<([A-Z_]+)>(.*?)</\1>", re.DOTALL)
LABELS = ["COORDINATION", "CORRECT_ARG", "CORRECT_HYPO", "CORRECT_INFORMATION", "CORRECT_REJECT_ARG", "CORRECT_REJECT_HYPO", "CORRECT_SUPPORT_ARG", "CORRECT_SUPPORT_HYPO", "INCORRECT_ARG", "INCORRECT_HYPO", "INCORRECT_REJECT_ARG", "INCORRECT_REJECT_HYPO", "INCORRECT_SUPPORT_ARG", "INCORRECT_SUPPORT_HYPO", "REASONING"]

def parse_file(path):
    """Return (plain_text, spans) where spans is a list of (start, end, label)
    covering the ENTIRE plain_text contiguously (label='O' for untagged text)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    spans = []
    plain_parts = []
    pos = 0
    cursor = 0  # position in plain_text being built

    for m in TAG_RE.finditer(raw):
        # untagged text before this tag
        if m.start() > pos:
            chunk = raw[pos:m.start()]
            if chunk:
                plain_parts.append(chunk)
                spans.append((cursor, cursor + len(chunk), "O"))
                cursor += len(chunk)
        label, content = m.group(1), m.group(2)
        if content:
            plain_parts.append(content)
            spans.append((cursor, cursor + len(content), label))
            cursor += len(content)
        pos = m.end()

    # trailing untagged text
    if pos < len(raw):
        chunk = raw[pos:]
        if chunk:
            plain_parts.append(chunk)
            spans.append((cursor, cursor + len(chunk), "O"))
            cursor += len(chunk)

    plain_text = "".join(plain_parts)
    return plain_text, spans


def label_at(spans, char_idx):
    # linear scan is fine for these file sizes; spans are sorted & contiguous
    for start, end, label in spans:
        if start <= char_idx < end:
            return label
    return "O"


def tokenize_with_offsets(text):
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def align_and_tokenize(plain1, spans1, plain2, spans2):
    """If the two plain texts differ slightly, align them with difflib so we
    still compare like-for-like tokens instead of crashing on an offset mismatch."""
    if plain1 == plain2:
        toks = tokenize_with_offsets(plain1)
        pairs = []
        for start, end, tok in toks:
            l1 = label_at(spans1, start)
            l2 = label_at(spans2, start)
            pairs.append((tok, l1, l2))
        return pairs, True

    # Fallback: align via difflib on whitespace tokens, only compare tokens
    # that both annotators actually share (equal opcodes).
    import difflib
    toks1 = tokenize_with_offsets(plain1)
    toks2 = tokenize_with_offsets(plain2)
    words1 = [t[2] for t in toks1]
    words2 = [t[2] for t in toks2]
    sm = difflib.SequenceMatcher(a=words1, b=words2, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                start1 = toks1[i1 + k][0]
                start2 = toks2[j1 + k][0]
                l1 = label_at(spans1, start1)
                l2 = label_at(spans2, start2)
                pairs.append((toks1[i1 + k][2], l1, l2))
    return pairs, False


def cohen_kappa(pairs_labels):
    """pairs_labels: list of (label1, label2). Manual Cohen's kappa (no sklearn dep)."""
    n = len(pairs_labels)
    if n == 0:
        return float("nan")
    agree = sum(1 for a, b in pairs_labels if a == b)
    po = agree / n

    labels = sorted(set(l for pair in pairs_labels for l in pair))
    c1 = Counter(a for a, b in pairs_labels)
    c2 = Counter(b for a, b in pairs_labels)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in labels)

    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def token_level_report(pairs):
    labels_pairs = [(l1, l2) for _, l1, l2 in pairs]
    n = len(labels_pairs)
    acc = sum(1 for a, b in labels_pairs if a == b) / n if n else float("nan")
    kappa = cohen_kappa(labels_pairs)

    # per-label precision/recall/F1, treating annotator1 as reference, annotator2 as hypothesis
    labels = sorted(set(l for pair in labels_pairs for l in pair))
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for a, b in labels_pairs:
        if a == b:
            tp[a] += 1
        else:
            fn[a] += 1  # annotator1 said 'a' but annotator2 didn't match
            fp[b] += 1  # annotator2 said 'b' but annotator1 didn't match

    results = defaultdict(float)
    results["tokens_compared"] = n
    results["accuracy"] = acc
    results["kappa"] = kappa
    print(f"Tokens compared: {n}")
    print(f"Raw agreement (accuracy): {acc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print()
    print(f"{'Label':30s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Support':>8s}")
    for l in labels:
        support = tp[l] + fn[l]
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        rec = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"{l:30s} {prec:7.3f} {rec:7.3f} {f1:7.3f} {support:8d}")

        results[f"{l}_precision"] = prec
        results[f"{l}_recall"] = rec
        results[f"{l}_f1"] = f1
        results[f"{l}_support"] = support
    
    return results


def span_level_report(spans1, spans2):
    """Exact-match span comparison: merge consecutive same-label O-spans aren't
    merged here since original spans already come from distinct tags, but we
    compare (start,end,label) tuples directly (excluding 'O')."""
    set1 = {(s, e, l) for s, e, l in spans1 if l != "O"}
    set2 = {(s, e, l) for s, e, l in spans2 if l != "O"}

    tp = set1 & set2
    fp = set2 - set1
    fn = set1 - set2

    prec = len(tp) / len(set2) if set2 else 0.0
    rec = len(tp) / len(set1) if set1 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    results = defaultdict(float)
    results["num_spans_anno1"] = len(set1)
    results["num_spans_anno2"] = len(set2)
    results["matching_spans"] = len(tp)
    results["precision"] = prec
    results["recall"] = rec
    results["f1"] = f1

    print(f"Exact-match spans -- annotator1: {len(set1)}, annotator2: {len(set2)}, matched: {len(tp)}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print()
    print("Per-label exact-match F1:")
    labels = sorted(set(l for _, _, l in set1) | set(l for _, _, l in set2))
    for l in labels:
        s1 = {(s, e) for s, e, ll in set1 if ll == l}
        s2 = {(s, e) for s, e, ll in set2 if ll == l}
        m = s1 & s2
        p = len(m) / len(s2) if s2 else 0.0
        r = len(m) / len(s1) if s1 else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"  {l:28s} P={p:.3f} R={r:.3f} F1={f:.3f}  (ann1={len(s1)}, ann2={len(s2)})")

        results[f"{l}_num_spans_anno1"] = len(set1)
        results[f"{l}_num_spans_anno2"] = len(set2)
        results[f"{l}_precision"] = prec
        results[f"{l}_recall"] = rec
        results[f"{l}_f1"] = f1
    
    return results


def compare_all_files(task: str):
    # Compare CMU
    task_path = f"annotation/multi/{task}"
    model_dirs = os.listdir(task_path)
    token_results_per_model = defaultdict(lambda: defaultdict(dict)) # model x instances x metrics 
    span_results_per_model = defaultdict(lambda: defaultdict(dict)) # model x instances x metrics

    num_insts_total = 0
    num_insts_model = 3

    token_kappa_list = []
    token_kappa_per_model = defaultdict(float)
    token_f1_per_label = defaultdict(float)
    for model_dir in model_dirs:
        if model_dir != ".DS_Store":
            anno_path = f"{task_path}/{model_dir}/annotated"
            anno_1_path = f"{task_path}/{model_dir}/annotated/annotator_1"
            anno_2_path = f"{task_path}/{model_dir}/annotated/annotator_2"
            
            for inst in os.listdir(anno_1_path):
                if inst != ".DS_Store":
                    num_insts_total += 1
                    
                    file1 = f"{anno_1_path}/{inst}"
                    file2 = f"{anno_2_path}/{inst}"

                    plain1, spans1 = parse_file(file1)
                    plain2, spans2 = parse_file(file2)

                    pairs, exact_match = align_and_tokenize(plain1, spans1, plain2, spans2)

                    if not exact_match:
                        print("NOTE: underlying plain texts differ between the two files; "
                            "aligned via difflib and only comparing shared tokens.\n")

                    print("=" * 60)
                    print("INSTANCE: ", inst)
                    print("TOKEN-LEVEL AGREEMENT")
                    print("=" * 60)
                    token_inst_results = token_level_report(pairs)
                    token_results_per_model[model_dir][inst.split(".txt")[0]] = token_inst_results
                    token_kappa_list.append(token_inst_results["kappa"])
                    token_kappa_per_model[model_dir] += token_inst_results["kappa"]

                    for lbl in LABELS:
                        token_f1_per_label[lbl] += token_inst_results[f"{lbl}_f1"]

                    print()
                    print("=" * 60)
                    print("SPAN-LEVEL AGREEMENT (exact boundary + label match)")
                    print("=" * 60)
                    span_results_per_model[model_dir][inst.split(".txt")[0]] = span_level_report(spans1, spans2)
    
    token_kappa_avg = round((sum(token_kappa_list)/num_insts_total)*100, 2)
    token_kappa_per_model_avg = {model: round((kappa/num_insts_model)*100, 2) for model,kappa in token_kappa_per_model.items()}
    token_f1_per_label_avg = {lbl: round((f1/num_insts_total)*100, 2) for lbl, f1 in token_f1_per_label.items()} 

    print("=" * 60)
    print("AVERAGE TOKEN-LEVEL AGREEMENT")
    print("=" * 60)
    print(f"AVG TOKEN-LEVEL KAPPA OVER ALL INSTANCES:", token_kappa_avg)
    print("HIGHEST TOKEN-LEVEL KAPPA OVER ALL INSTANCES:", round(max(token_kappa_list)*100, 2))
    print("LOWEST TOKEN-LEVEL KAPPA OVER ALL INSTANCES:", round(min(token_kappa_list)*100, 2))

    print("AVERAGE TOKEN-LEVEL KAPPA PER MODEL")
    pp(token_kappa_per_model_avg)

    print("AVERAGE TOKEN-LEVEL KAPPA PER LABEL")
    pp(token_f1_per_label_avg)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1")
    ap.add_argument("file2")
    args = ap.parse_args()

    plain1, spans1 = parse_file(args.file1)
    plain2, spans2 = parse_file(args.file2)

    pairs, exact_match = align_and_tokenize(plain1, spans1, plain2, spans2)

    if not exact_match:
        print("NOTE: underlying plain texts differ between the two files; "
              "aligned via difflib and only comparing shared tokens.\n")

    print("=" * 60)
    print("TOKEN-LEVEL AGREEMENT")
    print("=" * 60)
    token_level_report(pairs)

    print()
    print("=" * 60)
    print("SPAN-LEVEL AGREEMENT (exact boundary + label match)")
    print("=" * 60)
    span_level_report(spans1, spans2)


if __name__ == "__main__":
    compare_all_files(task = "cmu_booksum")
