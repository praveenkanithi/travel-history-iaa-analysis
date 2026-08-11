"""Integration test: loads the real pkanithi/travel-history-iaa-benchmark
dataset from the Hub. Requires network access and a Hugging Face login/token
with access to the (private) dataset.
"""

from data import CLINICIAN_IDS, load_notes


def test_load_notes_shape():
    case_reports, clinicians, llms = load_notes()

    assert len(case_reports) == 100
    assert set(clinicians.keys()) == set(CLINICIAN_IDS)
    assert len(llms) == 9

    uid, report = next(iter(case_reports.items()))
    assert isinstance(report, str) and len(report) > 0

    # spot check one clinician response has the expected fields
    some_clinician_responses = next(r for r in clinicians.values() if r)
    uid, response = next(iter(some_clinician_responses.items()))
    assert "has_international_travel" in response
    assert "entities" in response
