import copy
import pytest

from benchmark import ENC, category, history_chunks, merge_rankings, pack_memories, select_samples


def test_history_boundary_never_ingests_questions_answers_or_evidence_flags():
    s = {"question": "SECRET QUESTION", "answer": "SECRET GOLD", "answer_session_ids": ["s"],
         "haystack_session_ids": ["s"], "haystack_dates": ["2024/01/01 (Mon) 00:00"],
         "haystack_sessions": [[{"role": "user", "content": "Hello", "has_answer": True}]]}
    chunks = history_chunks(s)
    assert "SECRET" not in str(chunks)
    assert "has_answer" not in str(chunks)
    assert chunks[0]["messages"][0]["role"] == "user"


def test_context_budget_includes_headers_and_multibyte_text():
    records = [{"id": str(i), "content": "记忆 café 🚀 " * 200,
                "source_timestamp": "2024-01-01T00:00:00+00:00"} for i in range(5)]
    for budget in [1, 17, 100, 4096]:
        text, _ = pack_memories(records, budget)
        assert len(ENC.encode(text)) <= budget


@pytest.mark.parametrize("fragment", ["original information ", "记忆 café 🚀 的信息"])
def test_chunking_preserves_roles_and_all_text_despite_large_turn(fragment):
    content = fragment * 1000
    s = {"haystack_session_ids": ["a"], "haystack_dates": ["2024/01/01 (Mon) 00:00"],
         "haystack_sessions": [[{"role": "assistant", "content": content}]]}
    chunks = history_chunks(s, max_tokens=200)
    messages = [m for c in chunks for m in c["messages"]]
    assert all(m["role"] == "assistant" for m in messages)
    assert "".join(m["content"].split("\n", 1)[1] for m in messages) == content


def test_selection_is_independent_of_answers_and_balanced_in_prefix():
    from benchmark import CATEGORIES
    rows = [{"question_id": f"{c}-{i}" + ("_abs" if c == "abstention" else ""),
             "question_type": c if c != "abstention" else "single-session-user", "answer": i}
            for c in CATEGORIES for i in range(8)]
    selected = select_samples(rows, 2, 42)
    changed = copy.deepcopy(rows)
    for x in changed:
        x["answer"] = "changed"
    assert [x["question_id"] for x in selected] == [x["question_id"] for x in select_samples(changed, 2, 42)]
    assert {category(x) for x in selected[:7]} == set(CATEGORIES)


def test_strategy_fusion_ignores_incomparable_native_scores():
    a = [{"id": "a", "content": "a", "score": 0.001}]
    b = [{"id": "b", "content": "b", "score": 999}]
    result = merge_rankings({"semantic": a, "summary": b})
    assert result[0]["fusion_score"] == result[1]["fusion_score"]


def test_history_is_chronological_and_duplicate_source_sessions_remain_separate():
    s = {"haystack_session_ids": ["same", "same"],
         "haystack_dates": ["2024/02/01 (Thu) 00:00", "2024/01/01 (Mon) 00:00"],
         "haystack_sessions": [[{"role": "user", "content": "Updated address"}],
                              [{"role": "user", "content": "Old address"}]]}
    chunks = history_chunks(s)
    assert "Old address" in chunks[0]["messages"][0]["content"]
    assert chunks[0]["session"] != chunks[1]["session"]


def test_equal_facts_on_different_source_dates_are_not_collapsed():
    records = [{"id": "a", "content": "Visited the library", "source_timestamp": "2024-01-01"},
               {"id": "b", "content": "Visited the library", "source_timestamp": "2024-02-01"}]
    _, used = pack_memories(records + [records[0]], 4096)
    assert [r["id"] for r in used] == ["a", "b"]
