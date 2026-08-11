import math

from metrics import (
    binary_prf,
    build_matrix,
    categorical_macro_f1,
    encode_nominal,
    f1_from_counts,
    jaccard_offset,
    krippendorff_alpha,
    majority_binary,
    majority_categorical,
    match_spans,
    observed_agreement,
    presence_value_fn,
    span_consensus,
)


def test_krippendorff_alpha_perfect_agreement():
    rater_data = {
        "a": {"u1": {"v": "x"}, "u2": {"v": "y"}},
        "b": {"u1": {"v": "x"}, "u2": {"v": "y"}},
    }
    uids = ["u1", "u2"]
    value_fn = encode_nominal(rater_data, uids, ["a", "b"], "v")
    matrix = build_matrix(rater_data, uids, ["a", "b"], value_fn)
    assert krippendorff_alpha(matrix, "nominal") == 1.0


def test_krippendorff_alpha_missing_rater_uses_nan():
    rater_data = {"a": {"u1": {"v": "x"}}, "b": {}}
    matrix = build_matrix(rater_data, ["u1"], ["a", "b"], lambda r: 1.0)
    assert math.isnan(matrix[0, 1])


def test_observed_agreement():
    matrix = build_matrix(
        {"a": {"u1": {"v": 1}}, "b": {"u1": {"v": 1}}, "c": {"u1": {"v": 0}}},
        ["u1"], ["a", "b", "c"], lambda r: float(r["v"]),
    )
    # pairs: (a,b) agree, (a,c) disagree, (b,c) disagree -> 1/3
    assert observed_agreement(matrix) == 1 / 3


def test_binary_prf():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, True]
    precision, recall, f1 = binary_prf(y_true, y_pred)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_binary_prf_no_positives():
    precision, recall, f1 = binary_prf([False, False], [False, False])
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_categorical_macro_f1_perfect():
    y_true = ["a", "b", "a", "c"]
    assert categorical_macro_f1(y_true, y_true) == 1.0


def test_jaccard_offset():
    assert jaccard_offset(0, 10, 0, 10) == 1.0
    assert jaccard_offset(0, 10, 10, 20) == 0.0
    assert jaccard_offset(0, 10, 5, 15) == 5 / 15


def test_match_spans_exact_and_relaxed():
    ref = [{"start": 0, "end": 10}]
    pred_exact = [{"start": 0, "end": 10}]
    pred_shifted = [{"start": 2, "end": 12}]  # jaccard = 8/12 = 0.667 >= 0.5, not exact

    assert match_spans(ref, pred_exact, exact=True) == (1, 0, 0)
    assert match_spans(ref, pred_shifted, exact=True) == (0, 1, 1)
    assert match_spans(ref, pred_shifted, exact=False) == (1, 0, 0)


def test_f1_from_counts():
    assert f1_from_counts(1, 0, 0) == 1.0
    assert math.isnan(f1_from_counts(0, 0, 0))


def test_majority_binary():
    assert majority_binary([True, True, False]) is True
    assert majority_binary([True, False]) is None
    assert majority_binary([]) is None


def test_majority_categorical():
    assert majority_categorical(["a", "a", "b"]) == "a"
    assert majority_categorical(["a", "b"]) is None


def test_span_consensus_requires_min_votes():
    per_rater = [
        [{"start": 0, "end": 5}],
        [{"start": 1, "end": 6}],
        [],
    ]
    consensus = span_consensus(per_rater, min_votes=2)
    assert consensus == [{"start": 0, "end": 6}]

    consensus_strict = span_consensus(per_rater, min_votes=3)
    assert consensus_strict == []


def test_presence_value_fn():
    fn = presence_value_fn("destination")
    assert fn({"entities": [{"label": "destination"}]}) == 1.0
    assert fn({"entities": [{"label": "travel_duration"}]}) == 0.0
    assert fn({"entities": None}) == 0.0
