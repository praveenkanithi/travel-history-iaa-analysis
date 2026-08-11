"""Shared statistics: Krippendorff's alpha, precision/recall/F1, and span
matching. Used by both the clinician-only IAA analysis and the LLM-vs-majority
evaluation.
"""

from __future__ import annotations

import warnings
from collections import Counter
from itertools import combinations

import krippendorff
import numpy as np

JACCARD_THRESHOLD = 0.5


# ── Reliability matrices + Krippendorff's alpha ─────────────────────────────

def encode_nominal(rater_data, uids, rater_ids, var):
    """Build a value_fn mapping a variable's raw values to integer codes, shared
    across all raters/uids so krippendorff.alpha sees a single category space."""
    observed = sorted({
        rater_data[r][u][var]
        for u in uids for r in rater_ids
        if u in rater_data[r] and rater_data[r][u].get(var) is not None
    })
    code = {v: i for i, v in enumerate(observed)}

    def value_fn(response):
        v = response.get(var)
        return np.nan if v is None else float(code[v])

    return value_fn


def presence_value_fn(entity_label):
    """value_fn: 1.0 if the response has >=1 entity of this label, else 0.0."""

    def value_fn(response):
        entities = response.get("entities") or []
        return 1.0 if any(e.get("label") == entity_label for e in entities) else 0.0

    return value_fn


def build_matrix(rater_data, uids, rater_ids, value_fn):
    """(units x raters) matrix; NaN where a rater didn't annotate that unit."""
    matrix = np.full((len(uids), len(rater_ids)), np.nan)
    for i, uid in enumerate(uids):
        for j, rid in enumerate(rater_ids):
            resp = rater_data[rid].get(uid)
            if resp is not None:
                matrix[i, j] = value_fn(resp)
    return matrix


def krippendorff_alpha(matrix, level="nominal"):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(krippendorff.alpha(reliability_data=matrix.T, level_of_measurement=level))
    except (ValueError, AssertionError):
        return float("nan")


def observed_agreement(matrix):
    """Fraction of (rater-pair, unit) comparisons where both raters agree."""
    total = agree = 0
    for row in matrix:
        valid = row[~np.isnan(row)]
        if len(valid) < 2:
            continue
        for a, b in combinations(valid, 2):
            total += 1
            agree += int(a == b)
    return agree / total if total else float("nan")


# ── Precision / recall / F1 ──────────────────────────────────────────────────

def binary_prf(y_true, y_pred):
    """y_true / y_pred: lists of bool, same length."""
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def categorical_macro_f1(y_true, y_pred):
    """One-vs-rest F1 per observed class, averaged (macro)."""
    classes = sorted(set(y_true))
    if not classes:
        return float("nan")
    f1s = [binary_prf([t == c for t in y_true], [p == c for p in y_pred])[2] for c in classes]
    return sum(f1s) / len(f1s)


# ── Span matching ────────────────────────────────────────────────────────────

def jaccard_offset(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def match_spans(ref, pred, exact):
    """Greedy maximum-overlap matching between two span lists.
    exact=True  -> identical (start, end) required.
    exact=False -> Jaccard >= JACCARD_THRESHOLD required.
    Returns (tp, fp, fn)."""
    matched = set()
    tp = 0
    threshold = 1.0 if exact else JACCARD_THRESHOLD
    for r in ref:
        best_score, best_j = -1.0, -1
        for j, p in enumerate(pred):
            if j in matched:
                continue
            if exact:
                score = 1.0 if (r["start"], r["end"]) == (p["start"], p["end"]) else 0.0
            else:
                score = jaccard_offset(r["start"], r["end"], p["start"], p["end"])
            if score > best_score:
                best_score, best_j = score, j
        if best_j >= 0 and best_score >= threshold:
            tp += 1
            matched.add(best_j)
    return tp, len(pred) - len(matched), len(ref) - tp


def f1_from_counts(tp, fp, fn):
    if tp + fp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


# ── Majority-vote builders (used to construct the LLM reference) ────────────

def majority_binary(values):
    """values: list of bool. Strict majority (>50%); None on tie or empty."""
    if not values:
        return None
    n_true = sum(values)
    n_total = len(values)
    if n_true * 2 > n_total:
        return True
    if (n_total - n_true) * 2 > n_total:
        return False
    return None


def majority_categorical(values):
    """values: list of raw category values. Strict majority; None on tie or empty."""
    if not values:
        return None
    value, count = Counter(values).most_common(1)[0]
    return value if count * 2 > len(values) else None


def span_consensus(per_rater_spans, min_votes):
    """Cluster-based majority vote for spans of a single entity label.
    per_rater_spans: list of span-lists, one per rater (possibly empty).
    Overlapping spans are swept into contiguous clusters; each rater counts at
    most once per cluster. Returns list of {'start', 'end'} (union of members),
    keeping only clusters supported by >= min_votes distinct raters."""
    events = []
    for i, spans in enumerate(per_rater_spans):
        for s in spans:
            if s["end"] > s["start"]:
                events.append((s["start"], s["end"], i))
    if not events:
        return []
    events.sort()
    clusters = []  # [start, end, {rater_indices}]
    for st, en, ri in events:
        if not clusters or st >= clusters[-1][1]:
            clusters.append([st, en, {ri}])
        else:
            clusters[-1][1] = max(clusters[-1][1], en)
            clusters[-1][2].add(ri)
    return [{"start": cs, "end": ce} for cs, ce, raters in clusters if len(raters) >= min_votes]
