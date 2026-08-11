"""LLM evaluation against a clinician-majority reference.

The reference is built from the same clinicians used in the IAA analysis
(strict majority vote for binary/categorical fields, cluster-based majority
vote for entity spans). Each LLM is then scored against that reference with:
  - label F1 for has_international_travel, the 8 categorical fields, and
    5 entity-presence indicators
  - span boundary F1 (exact + relaxed) for the 5 entity types

No LLM ensemble/consortium is built here -- each of the 9 models is scored
individually against the clinician reference.
"""

from __future__ import annotations

from data import BINARY_VAR, CATEGORICAL_VARS, ENTITY_TYPES, TRAVEL_CONDITIONAL_ENTITIES
from metrics import (
    binary_prf,
    categorical_macro_f1,
    f1_from_counts,
    majority_binary,
    majority_categorical,
    match_spans,
    span_consensus,
)


def build_majority_reference(clinicians, uids, clinician_ids):
    """{uid: {has_international_travel, <categorical vars>, entities}}."""
    reference = {}
    for uid in uids:
        responses = [clinicians[cid][uid] for cid in clinician_ids if uid in clinicians[cid]]
        if len(responses) < 2:
            continue
        min_votes = len(responses) // 2 + 1

        travel = majority_binary([
            bool(r[BINARY_VAR]) for r in responses if r.get(BINARY_VAR) is not None
        ])

        categorical = {
            var: majority_categorical([r[var] for r in responses if r.get(var) is not None])
            for var in CATEGORICAL_VARS
        }

        labels = {e["label"] for r in responses for e in (r.get("entities") or [])}
        entities = []
        for label in labels:
            per_rater = [[e for e in (r.get("entities") or []) if e["label"] == label] for r in responses]
            for span in span_consensus(per_rater, min_votes):
                entities.append({"label": label, **span})

        reference[uid] = {BINARY_VAR: travel, **categorical, "entities": entities}
    return reference


def _paired_values(llm_responses, reference, uids, value_fn):
    """value_fn(response) -> value or None. Returns (y_true, y_pred), skipping
    uids where either side is missing/None."""
    y_true, y_pred = [], []
    for uid in uids:
        ref = reference.get(uid)
        pred = llm_responses.get(uid)
        if ref is None or pred is None:
            continue
        t, p = value_fn(ref), value_fn(pred)
        if t is None or p is None:
            continue
        y_true.append(t)
        y_pred.append(p)
    return y_true, y_pred


def binary_label_f1(llm_responses, reference, uids, var):
    y_true, y_pred = _paired_values(llm_responses, reference, uids, lambda r: r.get(var))
    if not y_true:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "n": 0}
    p, r, f1 = binary_prf([bool(v) for v in y_true], [bool(v) for v in y_pred])
    return {"precision": p, "recall": r, "f1": f1, "n": len(y_true)}


def categorical_label_f1(llm_responses, reference, uids, var):
    y_true, y_pred = _paired_values(llm_responses, reference, uids, lambda r: r.get(var))
    if not y_true:
        return {"f1": float("nan"), "n": 0}
    return {"f1": categorical_macro_f1(y_true, y_pred), "n": len(y_true)}


def presence_label_f1(llm_responses, reference, uids, etype):
    def value_fn(r):
        return any(e["label"] == etype for e in (r.get("entities") or []))

    y_true, y_pred = _paired_values(llm_responses, reference, uids, value_fn)
    if not y_true:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "n": 0}
    p, r, f1 = binary_prf(y_true, y_pred)
    return {"precision": p, "recall": r, "f1": f1, "n": len(y_true)}


def span_boundary_f1(llm_responses, reference, uids, etype):
    exact_tp = exact_fp = exact_fn = 0
    relax_tp = relax_fp = relax_fn = 0
    n_notes = 0
    for uid in uids:
        ref = reference.get(uid)
        pred = llm_responses.get(uid)
        if ref is None or pred is None:
            continue
        ref_spans = [e for e in ref.get("entities", []) if e["label"] == etype]
        pred_spans = [e for e in (pred.get("entities") or []) if e["label"] == etype]
        n_notes += 1
        tp, fp, fn = match_spans(ref_spans, pred_spans, exact=True)
        exact_tp += tp
        exact_fp += fp
        exact_fn += fn
        tp, fp, fn = match_spans(ref_spans, pred_spans, exact=False)
        relax_tp += tp
        relax_fp += fp
        relax_fn += fn
    return {
        "exact_f1": f1_from_counts(exact_tp, exact_fp, exact_fn),
        "relaxed_f1": f1_from_counts(relax_tp, relax_fp, relax_fn),
        "n_notes": n_notes,
    }


def evaluate_llm(llm_responses, reference, all_uids, travel_uids):
    """Full label F1 + span boundary F1 for one LLM against the reference."""
    label_f1 = {BINARY_VAR: binary_label_f1(llm_responses, reference, all_uids, BINARY_VAR)}

    for var in CATEGORICAL_VARS:
        label_f1[var] = categorical_label_f1(llm_responses, reference, travel_uids, var)

    span_boundary = {}
    for etype in ENTITY_TYPES:
        conditional = etype in TRAVEL_CONDITIONAL_ENTITIES
        uids = travel_uids if conditional else all_uids
        label_f1[f"{etype}_presence"] = presence_label_f1(llm_responses, reference, uids, etype)
        span_boundary[etype] = span_boundary_f1(llm_responses, reference, uids, etype)

    return {"label_f1": label_f1, "span_boundary_f1": span_boundary}


def evaluate_all_llms(llms, reference, all_uids, travel_uids):
    return {lid: evaluate_llm(responses, reference, all_uids, travel_uids) for lid, responses in llms.items()}
