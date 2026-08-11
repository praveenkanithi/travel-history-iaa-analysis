from fixtures import synthetic_clinicians

import clinician_iaa as iaa


def setup():
    clinician_ids, clinicians = synthetic_clinicians()
    all_uids = iaa.notes_with_min_raters(clinicians, min_raters=2)
    travel_uids = iaa.travel_present_uids(clinicians, all_uids)
    return clinician_ids, clinicians, all_uids, travel_uids


def test_notes_with_min_raters():
    clinician_ids, clinicians = synthetic_clinicians()
    uids = iaa.notes_with_min_raters(clinicians, min_raters=2)
    assert set(uids) == {"u_travel", "u_no_travel", "u_span_disagreement"}


def test_travel_present_uids():
    clinician_ids, clinicians, all_uids, travel_uids = setup()
    assert set(travel_uids) == {"u_travel", "u_span_disagreement"}


def test_bucket1_travel_presence_unanimous_pair():
    clinician_ids, clinicians, all_uids, travel_uids = setup()
    result = iaa.bucket1_travel_presence(clinicians, all_uids, clinician_ids)
    assert result["n_notes"] == 3
    # u_travel and u_span_disagreement unanimous True, u_no_travel unanimous False -> perfect agreement
    assert result["observed_agreement"] == 1.0


def test_bucket3_span_presence_conditional_split():
    clinician_ids, clinicians, all_uids, travel_uids = setup()
    rows = iaa.bucket3_span_presence(clinicians, all_uids, travel_uids, clinician_ids)
    destination_row = next(r for r in rows if r["entity_type"] == "destination")
    assert destination_row["conditional"] is True
    assert destination_row["n_notes"] == len(travel_uids)

    onset_entity_row = next(r for r in rows if r["entity_type"] == "time_since_first_symptom_at_presentation")
    assert onset_entity_row["conditional"] is False
    assert onset_entity_row["n_notes"] == len(all_uids)


def test_bucket4_span_boundaries_exact_vs_relaxed():
    clinician_ids, clinicians, all_uids, travel_uids = setup()
    rows = iaa.bucket4_span_boundaries(clinicians, all_uids, travel_uids, clinician_ids)
    destination_row = next(r for r in rows if r["entity_type"] == "destination")
    # u_span_disagreement has a shifted-but-overlapping span and a missing span,
    # so relaxed F1 must be >= exact F1 (relaxed matching is strictly easier).
    assert destination_row["relaxed_f1"] >= destination_row["exact_f1"]
    # u_travel is unanimous (all 5 clinicians agree exactly) -> some pairs are perfect
    assert destination_row["n_pair_note_comparisons"] > 0


def test_leave_one_out_covers_every_clinician_and_variable():
    clinician_ids, clinicians, all_uids, travel_uids = setup()
    loo_rows = iaa.leave_one_out_alpha(clinicians, all_uids, travel_uids, clinician_ids)
    assert {row["held_out"] for row in loo_rows} == set(clinician_ids)
    variables = {v["variable"] for v in loo_rows[0]["variables"]}
    assert "has_international_travel" in variables
    assert "destination_presence" in variables
    for row in loo_rows:
        assert len(row["variables"]) == len(loo_rows[0]["variables"])
