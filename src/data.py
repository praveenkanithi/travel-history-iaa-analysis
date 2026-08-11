"""Load the pkanithi/travel-history-iaa-benchmark dataset from the Hub and
parse it into plain Python structures used by the rest of the analysis.
"""

from __future__ import annotations

import json

from datasets import load_dataset

DATASET_ID = "pkanithi/travel-history-iaa-benchmark"

CLINICIAN_IDS = [f"clinician_{i}" for i in range(1, 6)]

BINARY_VAR = "has_international_travel"

CATEGORICAL_VARS = [
    "symptom_onset_relative_to_return",
    "exposure_food_water",
    "exposure_environmental",
    "exposure_bites_animal",
    "exposure_sick_contacts",
    "exposure_sexual_contact",
    "exposure_healthcare_abroad",
    "exposure_percutaneous",
]

ENTITY_TYPES = [
    "destination",
    "travel_duration",
    "time_since_return",
    "time_between_return_and_first_symptom",
    "time_since_first_symptom_at_presentation",
]

# Entities that only make sense once travel has happened; the fifth entity type
# (symptom duration at presentation) can occur with or without travel.
TRAVEL_CONDITIONAL_ENTITIES = {
    "destination",
    "travel_duration",
    "time_since_return",
    "time_between_return_and_first_symptom",
}


def load_notes():
    """Return (case_reports, clinicians, llms).

    case_reports : {uid: case report text}
    clinicians   : {clinician_id: {uid: parsed response dict}}
    llms         : {llm_id: {uid: parsed response dict}}

    Each parsed response dict has keys: has_international_travel,
    symptom_onset_relative_to_return, the 7 exposure_* fields, entities
    (list of {label, text, start, end}), and comment.
    """
    ds = load_dataset(DATASET_ID)["train"]
    llm_ids = [c for c in ds.column_names if c.startswith("llm_")]

    case_reports = {}
    clinicians = {cid: {} for cid in CLINICIAN_IDS}
    llms = {lid: {} for lid in llm_ids}

    for row in ds:
        uid = row["uid"]
        case_reports[uid] = row["case_report"]
        for cid in CLINICIAN_IDS:
            raw = row[cid]
            if raw is not None:
                clinicians[cid][uid] = json.loads(raw)
        for lid in llm_ids:
            raw = row[lid]
            if raw is not None:
                llms[lid][uid] = json.loads(raw)

    return case_reports, clinicians, llms
