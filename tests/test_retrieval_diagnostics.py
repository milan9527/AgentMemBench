import pytest

from diagnose_retrieval import (
    all_labels, digest, manual_labels, metrics, save, source_verified_corrections,
    validate_labels, visible_blocks,
)


def label(*facts):
    return {"supports": [{"fact_id": x, "quote": "evidence"} for x in facts],
            "uncertain_fact_ids": []}


def test_context_truncation_does_not_expose_hidden_evidence():
    record = {"id": "x", "content": "irrelevant preamble; critical hidden fact"}
    row = {"context": "[Memory 1]\nirrelevant preamble;",
           "used_memories": [{**record, "truncated": True}]}
    blocks = visible_blocks(row)
    assert blocks[0]["text"] == "irrelevant preamble;"
    assert blocks[0]["truncated"]
    assert "hidden" not in blocks[0]["text"]


def test_context_reconstruction_rejects_divergent_artifacts():
    with pytest.raises(ValueError):
        visible_blocks({"context": "[Memory 1]\nchanged",
                        "used_memories": [{"id": "x", "content": "original"}]})


def test_duplicates_and_short_results_do_not_inflate_scores():
    scores = metrics(["a", "a"], {"a": label("f1")}, {"f1", "f2"}, 5, 1)
    assert scores["fact_recall"] == .5
    assert scores["precision"] == .2
    assert scores["precision_returned"] == .5
    assert scores["ndcg_pooled"] == 1
    assert scores["full_fact_coverage"] == 0


def test_quote_must_be_in_the_candidate_not_only_the_rubric():
    records = [{"id": "a", "text": "nothing relevant"}]
    gold = {"facts": [{"id": "f1"}]}
    response = '{"items":[{"id":"a","supports":[{"fact_id":"f1","quote":"answer from rubric"}],"uncertain_fact_ids":[]}]}'
    with pytest.raises(AssertionError, match="quote not present"):
        validate_labels(response, records, gold)


def test_missing_labels_are_execution_failure_not_negative_relevance():
    with pytest.raises(AssertionError, match="Missing"):
        validate_labels('{"items":[]}', [{"id": "a", "text": "x"}],
                        {"facts": [{"id": "f1"}]})


def test_multi_fact_recall_and_shared_ideal_ranking():
    labels = {"a": label("f1"), "b": label("f2"), "c": label()}
    scores = metrics(["c", "b", "a"], labels, {"f1", "f2"}, 3, 2)
    assert scores["fact_recall"] == 1
    assert scores["mrr"] == .5
    assert 0 < scores["ndcg_pooled"] < 1


def test_source_correction_requires_full_message_and_original_date():
    source = {"full_message": "I own a Suica card.", "quote": "Suica card",
              "date": "2023/05/22 (Mon) 23:46", "session_id": "s", "turn_index": 0}
    gold = {"q": {"facts": [{"id": "card", "sources": [source]}]}}
    records = {"a": {"text": "Try a Suica card."},
               "b": {"text": "I own a Suica card."},
               "c": {"text": "[Conversation date: 2023/05/22 (Mon) 23:46]\nI own a Suica card."}}
    labels = {"q": {k: label() for k in records}}
    changes = source_verified_corrections(gold, {"q": {"records": records}}, labels)
    assert [(x["record_id"], x["fact_id"]) for x in changes] == [("c", "card")]


def test_direct_review_cannot_silently_apply_to_changed_text(tmp_path):
    save(tmp_path / "manual_labels.json", [{
        "question_id": "q", "record_id": "r", "text_sha256": digest("reviewed text"),
        "label": label(),
    }])
    with pytest.raises(AssertionError, match="changed text"):
        manual_labels(tmp_path, {"q": {"records": {"r": {"text": "different text"}}}})


def test_direct_adjudication_takes_precedence_over_model_review(tmp_path):
    save(tmp_path / "labels/q/batch.json", {"labels": {"r": label("f")}})
    save(tmp_path / "review_labels/q/r.json", {"record_id": "r", "label": label("f")})
    save(tmp_path / "manual_labels.json", [{
        "question_id": "q", "record_id": "r", "text_sha256": digest("~4 years"),
        "label": label(),
    }])
    result = all_labels(tmp_path, {"q": {"records": {"r": {"text": "~4 years"}}}})
    assert result["q"]["r"]["supports"] == []
    initial = all_labels(tmp_path, {"q": {"records": {"r": {"text": "~4 years"}}}},
                         include_review=False)
    assert initial["q"]["r"]["supports"][0]["fact_id"] == "f"
