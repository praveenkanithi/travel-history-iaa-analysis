from fixtures import synthetic_clinicians, synthetic_llms

import clinician_iaa as iaa
import llm_eval


def setup():
    clinician_ids, clinicians = synthetic_clinicians()
    llms = synthetic_llms()
    all_uids = iaa.notes_with_min_raters(clinicians, min_raters=2)
    reference = llm_eval.build_majority_reference(clinicians, all_uids, clinician_ids)
    travel_uids = [uid for uid, ref in reference.items() if ref.get(iaa.BINARY_VAR) is True]
    return clinicians, llms, all_uids, reference, travel_uids


def test_majority_reference_matches_unanimous_clinicians():
    clinicians, llms, all_uids, reference, travel_uids = setup()
    assert reference["u_travel"]["has_international_travel"] is True
    assert reference["u_no_travel"]["has_international_travel"] is False
    dest_spans = [e for e in reference["u_travel"]["entities"] if e["label"] == "destination"]
    assert len(dest_spans) == 1


def test_perfect_llm_scores_higher_than_noisy_llm():
    clinicians, llms, all_uids, reference, travel_uids = setup()
    result_perfect = llm_eval.evaluate_llm(llms["llm_perfect"], reference, all_uids, travel_uids)
    result_noisy = llm_eval.evaluate_llm(llms["llm_noisy"], reference, all_uids, travel_uids)

    f1_perfect = result_perfect["label_f1"]["has_international_travel"]["f1"]
    f1_noisy = result_noisy["label_f1"]["has_international_travel"]["f1"]
    assert f1_perfect == 1.0
    assert f1_noisy < f1_perfect


def test_span_boundary_f1_exact_match():
    clinicians, llms, all_uids, reference, travel_uids = setup()
    result = llm_eval.evaluate_llm(llms["llm_perfect"], reference, all_uids, travel_uids)
    destination = result["span_boundary_f1"]["destination"]
    # llm_perfect's (10,15) span matches u_travel's reference exactly, but on
    # u_span_disagreement the majority-consensus span is union-merged to
    # (10,16) (3 clinicians at (10,15), 1 overlapping at (10,16)), so that
    # note only matches under relaxed (Jaccard) scoring, not exact.
    assert destination["exact_f1"] == 0.5
    assert destination["relaxed_f1"] == 1.0


def test_evaluate_all_llms_covers_every_model():
    clinicians, llms, all_uids, reference, travel_uids = setup()
    results = llm_eval.evaluate_all_llms(llms, reference, all_uids, travel_uids)
    assert set(results.keys()) == set(llms.keys())
    for lid in llms:
        assert "label_f1" in results[lid]
        assert "span_boundary_f1" in results[lid]
