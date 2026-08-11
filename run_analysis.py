#!/usr/bin/env python3
"""Run the full inter-annotator-agreement + LLM evaluation and print results
to the terminal, then save the same results (with a timestamp) as JSON under
results/.

Usage:
    python run_analysis.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tabulate import tabulate

import clinician_iaa as iaa
import llm_eval
from data import CATEGORICAL_VARS, CLINICIAN_IDS, ENTITY_TYPES, load_notes

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def print_clinician_tables(bucket1, bucket2, bucket3, bucket4):
    print("\n== Table 1: Travel presence (Krippendorff alpha) ==")
    print(tabulate([[bucket1["variable"], fmt(bucket1["alpha"]), fmt(bucket1["observed_agreement"]), bucket1["n_notes"]]],
                    headers=["variable", "alpha", "observed_agreement", "n_notes"]))

    print("\n== Table 2: Categorical fields, travel-present subset (Krippendorff alpha) ==")
    print(tabulate(
        [[r["variable"], fmt(r["alpha"]), fmt(r["observed_agreement"]), r["n_notes"]] for r in bucket2],
        headers=["variable", "alpha", "observed_agreement", "n_notes"],
    ))

    print("\n== Table 3: Entity-presence agreement (Krippendorff alpha) ==")
    print(tabulate(
        [[r["entity_type"], r["conditional"], fmt(r["alpha"]), fmt(r["observed_agreement"]), r["n_notes"]] for r in bucket3],
        headers=["entity_type", "travel_conditional", "alpha", "observed_agreement", "n_notes"],
    ))

    print("\n== Table 4: Pairwise span boundary F1 among clinicians ==")
    print(tabulate(
        [[r["entity_type"], r["conditional"], fmt(r["exact_f1"]), fmt(r["relaxed_f1"]), r["n_pair_note_comparisons"]] for r in bucket4],
        headers=["entity_type", "travel_conditional", "exact_f1", "relaxed_f1 (Jaccard>=0.5)", "n_pair_comparisons"],
    ))


def print_loo_table(loo_rows, bucket1, bucket2, bucket3):
    full_alpha = {bucket1["variable"]: bucket1["alpha"]}
    full_alpha.update({r["variable"]: r["alpha"] for r in bucket2})
    full_alpha.update({f"{r['entity_type']}_presence": r["alpha"] for r in bucket3})

    variables = [row["variable"] for row in loo_rows[0]["variables"]]
    held_out_ids = [row["held_out"] for row in loo_rows]

    table = []
    for var in variables:
        row = [var, fmt(full_alpha.get(var))]
        for loo_row in loo_rows:
            alpha = next(v["alpha"] for v in loo_row["variables"] if v["variable"] == var)
            row.append(fmt(alpha))
        table.append(row)

    print("\n== Table 5: Leave-one-out alpha (each column = alpha with that clinician removed) ==")
    print(tabulate(table, headers=["variable", "alpha (all 5)"] + held_out_ids))


def print_llm_tables(llm_results):
    llm_ids = sorted(llm_results.keys())
    variables = [iaa.BINARY_VAR] + CATEGORICAL_VARS + [f"{e}_presence" for e in ENTITY_TYPES]

    print("\n== Table 6: LLM label F1 vs. clinician-majority reference ==")
    table = []
    for var in variables:
        row = [var]
        for lid in llm_ids:
            row.append(fmt(llm_results[lid]["label_f1"][var]["f1"]))
        table.append(row)
    print(tabulate(table, headers=["variable"] + llm_ids))

    print("\n== Table 7: LLM span boundary F1 (exact) ==")
    table = []
    for etype in ENTITY_TYPES:
        row = [etype] + [fmt(llm_results[lid]["span_boundary_f1"][etype]["exact_f1"]) for lid in llm_ids]
        table.append(row)
    print(tabulate(table, headers=["entity_type"] + llm_ids))

    print("\n== Table 8: LLM span boundary F1 (relaxed, Jaccard>=0.5) ==")
    table = []
    for etype in ENTITY_TYPES:
        row = [etype] + [fmt(llm_results[lid]["span_boundary_f1"][etype]["relaxed_f1"]) for lid in llm_ids]
        table.append(row)
    print(tabulate(table, headers=["entity_type"] + llm_ids))


def sanitize_nans(obj):
    """Recursively replace float('nan') with None so json.dumps produces
    strict, portable JSON (no bare NaN tokens)."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nans(v) for v in obj]
    return obj


def main():
    print("Loading pkanithi/travel-history-iaa-benchmark ...")
    case_reports, clinicians, llms = load_notes()

    all_uids = iaa.notes_with_min_raters(clinicians, min_raters=2)
    travel_uids = iaa.travel_present_uids(clinicians, all_uids)
    print(f"Notes: {len(all_uids)} total, {len(travel_uids)} majority-travel-present")
    print(f"Clinicians: {CLINICIAN_IDS}")
    print(f"LLMs: {sorted(llms.keys())}")

    # ── Clinician-only IAA ──────────────────────────────────────────────────
    bucket1 = iaa.bucket1_travel_presence(clinicians, all_uids, CLINICIAN_IDS)
    bucket2 = iaa.bucket2_categorical(clinicians, travel_uids, CLINICIAN_IDS)
    bucket3 = iaa.bucket3_span_presence(clinicians, all_uids, travel_uids, CLINICIAN_IDS)
    bucket4 = iaa.bucket4_span_boundaries(clinicians, all_uids, travel_uids, CLINICIAN_IDS)
    loo_rows = iaa.leave_one_out_alpha(clinicians, all_uids, travel_uids, CLINICIAN_IDS)

    print_clinician_tables(bucket1, bucket2, bucket3, bucket4)
    print_loo_table(loo_rows, bucket1, bucket2, bucket3)

    # ── LLM evaluation vs. clinician-majority reference ─────────────────────
    reference = llm_eval.build_majority_reference(clinicians, all_uids, CLINICIAN_IDS)
    reference_travel_uids = [uid for uid, ref in reference.items() if ref.get(iaa.BINARY_VAR) is True]
    llm_results = llm_eval.evaluate_all_llms(llms, reference, all_uids, reference_travel_uids)

    print_llm_tables(llm_results)

    # ── Save results JSON ────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "pkanithi/travel-history-iaa-benchmark",
        "n_notes": len(all_uids),
        "n_travel_present_notes": len(travel_uids),
        "clinicians": CLINICIAN_IDS,
        "llms": sorted(llms.keys()),
        "clinician_iaa": {
            "travel_presence": bucket1,
            "categorical": bucket2,
            "span_presence": bucket3,
            "span_boundaries": bucket4,
            "leave_one_out": loo_rows,
        },
        "llm_evaluation": llm_results,
    }
    with open(out_path, "w") as f:
        json.dump(sanitize_nans(results), f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
