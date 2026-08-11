"""Clinician-only inter-annotator agreement.

Four variable groups, each scored with Krippendorff's alpha (nominal) and
observed pairwise agreement:
  1. has_international_travel      -- all notes with >=2 clinicians
  2. 8 categorical fields          -- travel-present subset (majority gate)
  3. 5 entity-presence indicators  -- travel-present subset, except
                                       time_since_first_symptom_at_presentation
  4. 5 entity boundary F1 (exact + relaxed) -- same population as (3),
                                                 pairwise among clinicians

Also includes a leave-one-out (LOO) analysis: alpha recomputed for every
variable with each clinician held out in turn, to see how much overall
agreement depends on any single rater.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from data import BINARY_VAR, CATEGORICAL_VARS, ENTITY_TYPES, TRAVEL_CONDITIONAL_ENTITIES
from metrics import (
    build_matrix,
    encode_nominal,
    f1_from_counts,
    krippendorff_alpha,
    majority_binary,
    match_spans,
    observed_agreement,
    presence_value_fn,
)


def notes_with_min_raters(clinicians, min_raters=2):
    counts = {}
    for responses in clinicians.values():
        for uid in responses:
            counts[uid] = counts.get(uid, 0) + 1
    return sorted(uid for uid, n in counts.items() if n >= min_raters)


def travel_alpha_value_fn(response):
    v = response.get(BINARY_VAR)
    return np.nan if v is None else float(bool(v))


def travel_present_uids(clinicians, uids):
    """Strict-majority gate on has_international_travel. Ties are excluded."""
    present = []
    for uid in uids:
        votes = [
            bool(clinicians[cid][uid][BINARY_VAR])
            for cid in clinicians
            if uid in clinicians[cid] and clinicians[cid][uid].get(BINARY_VAR) is not None
        ]
        if majority_binary(votes) is True:
            present.append(uid)
    return present


def bucket1_travel_presence(clinicians, uids, clinician_ids):
    matrix = build_matrix(clinicians, uids, clinician_ids, travel_alpha_value_fn)
    return {
        "variable": BINARY_VAR,
        "alpha": krippendorff_alpha(matrix, "nominal"),
        "observed_agreement": observed_agreement(matrix),
        "n_notes": len(uids),
    }


def bucket2_categorical(clinicians, travel_uids, clinician_ids):
    rows = []
    for var in CATEGORICAL_VARS:
        value_fn = encode_nominal(clinicians, travel_uids, clinician_ids, var)
        matrix = build_matrix(clinicians, travel_uids, clinician_ids, value_fn)
        rows.append({
            "variable": var,
            "alpha": krippendorff_alpha(matrix, "nominal"),
            "observed_agreement": observed_agreement(matrix),
            "n_notes": len(travel_uids),
        })
    return rows


def bucket3_span_presence(clinicians, all_uids, travel_uids, clinician_ids):
    rows = []
    for etype in ENTITY_TYPES:
        conditional = etype in TRAVEL_CONDITIONAL_ENTITIES
        uids = travel_uids if conditional else all_uids
        matrix = build_matrix(clinicians, uids, clinician_ids, presence_value_fn(etype))
        rows.append({
            "entity_type": etype,
            "conditional": conditional,
            "alpha": krippendorff_alpha(matrix, "nominal"),
            "observed_agreement": observed_agreement(matrix),
            "n_notes": len(uids),
        })
    return rows


def bucket4_span_boundaries(clinicians, all_uids, travel_uids, clinician_ids):
    rows = []
    for etype in ENTITY_TYPES:
        conditional = etype in TRAVEL_CONDITIONAL_ENTITIES
        uids = travel_uids if conditional else all_uids
        exact_tp = exact_fp = exact_fn = 0
        relax_tp = relax_fp = relax_fn = 0
        n_pairs = 0
        for uid in uids:
            present = [cid for cid in clinician_ids if uid in clinicians[cid]]
            for a, b in combinations(present, 2):
                spans_a = [e for e in (clinicians[a][uid].get("entities") or []) if e["label"] == etype]
                spans_b = [e for e in (clinicians[b][uid].get("entities") or []) if e["label"] == etype]
                n_pairs += 1
                for ref, pred in [(spans_a, spans_b), (spans_b, spans_a)]:
                    tp, fp, fn = match_spans(ref, pred, exact=True)
                    exact_tp += tp
                    exact_fp += fp
                    exact_fn += fn
                    tp, fp, fn = match_spans(ref, pred, exact=False)
                    relax_tp += tp
                    relax_fp += fp
                    relax_fn += fn
        rows.append({
            "entity_type": etype,
            "conditional": conditional,
            "exact_f1": f1_from_counts(exact_tp, exact_fp, exact_fn),
            "relaxed_f1": f1_from_counts(relax_tp, relax_fp, relax_fn),
            "n_pair_note_comparisons": n_pairs,
        })
    return rows


def leave_one_out_alpha(clinicians, all_uids, travel_uids, clinician_ids):
    """For each clinician removed in turn, recompute alpha for every variable
    using the remaining 4 clinicians."""
    rows = []
    for held_out in clinician_ids:
        remaining = [c for c in clinician_ids if c != held_out]

        travel_matrix = build_matrix(clinicians, all_uids, remaining, travel_alpha_value_fn)
        variables = [{
            "variable": BINARY_VAR,
            "alpha": krippendorff_alpha(travel_matrix, "nominal"),
        }]

        for var in CATEGORICAL_VARS:
            value_fn = encode_nominal(clinicians, travel_uids, remaining, var)
            matrix = build_matrix(clinicians, travel_uids, remaining, value_fn)
            variables.append({"variable": var, "alpha": krippendorff_alpha(matrix, "nominal")})

        for etype in ENTITY_TYPES:
            conditional = etype in TRAVEL_CONDITIONAL_ENTITIES
            uids = travel_uids if conditional else all_uids
            matrix = build_matrix(clinicians, uids, remaining, presence_value_fn(etype))
            variables.append({
                "variable": f"{etype}_presence",
                "alpha": krippendorff_alpha(matrix, "nominal"),
            })

        rows.append({"held_out": held_out, "variables": variables})
    return rows
