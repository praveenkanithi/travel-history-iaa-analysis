"""Small synthetic clinician/LLM datasets shared across tests."""


def make_response(travel, onset=None, entities=None, **exposures):
    base = {
        "has_international_travel": travel,
        "symptom_onset_relative_to_return": onset,
        "exposure_food_water": "not_mentioned",
        "exposure_environmental": "not_mentioned",
        "exposure_bites_animal": "not_mentioned",
        "exposure_sick_contacts": "not_mentioned",
        "exposure_sexual_contact": "not_mentioned",
        "exposure_healthcare_abroad": "not_mentioned",
        "exposure_percutaneous": "not_mentioned",
        "entities": entities or [],
    }
    base.update(exposures)
    return base


def synthetic_clinicians():
    """5 clinicians, 3 notes: unanimous travel, unanimous no-travel, and a
    travel-present note where one clinician's span boundary is shifted and
    another found no span at all (for span presence/boundary F1 coverage)."""
    clinician_ids = [f"clinician_{i}" for i in range(1, 6)]

    dest_span = {"label": "destination", "text": "Kenya", "start": 10, "end": 15}
    dest_span_shifted = {"label": "destination", "text": "Kenyaa", "start": 10, "end": 16}

    responses = {
        "u_travel": [
            make_response(True, "after_return", [dest_span], exposure_food_water="yes")
            for _ in clinician_ids
        ],
        "u_no_travel": [make_response(False) for _ in clinician_ids],
        "u_span_disagreement": [
            make_response(True, "before_return", [dest_span]),
            make_response(True, "before_return", [dest_span]),
            make_response(True, "before_return", [dest_span_shifted]),
            make_response(True, "before_return", []),
            make_response(True, "before_return", [dest_span]),
        ],
    }

    clinicians = {cid: {} for cid in clinician_ids}
    for uid, per_clinician in responses.items():
        for cid, resp in zip(clinician_ids, per_clinician):
            clinicians[cid][uid] = resp

    return clinician_ids, clinicians


def synthetic_llms():
    """2 toy LLMs scored against the same 3 uids as synthetic_clinicians()."""
    dest_span = {"label": "destination", "text": "Kenya", "start": 10, "end": 15}
    llms = {
        "llm_perfect": {
            "u_travel": make_response(True, "after_return", [dest_span], exposure_food_water="yes"),
            "u_no_travel": make_response(False),
            "u_span_disagreement": make_response(True, "before_return", [dest_span]),
        },
        "llm_noisy": {
            "u_travel": make_response(False),
            "u_no_travel": make_response(False),
            "u_span_disagreement": make_response(True, "after_return", []),
        },
    }
    return llms
