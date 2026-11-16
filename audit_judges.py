"""Repeat judging for every fixed answer; never overwrite the primary scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time

from benchmark import ARMS, ROOT, Bedrock, Journal, digest, judge_prompt, read, write


def audit(run):
    folder = ROOT / "runs" / run
    cfg = read(folder / "config.json")
    samples = {s["question_id"]: s for s in read(folder / "samples.json")}
    tasks = []
    for arm in ARMS:
        for qid, sample in samples.items():
            row = read(folder / "results" / arm / f"{qid}.json")
            assert row["status"] == "ok"
            tasks.append((arm, sample, row))
    bedrock = Bedrock(cfg, Journal(folder / "usage.jsonl"))

    def score(task):
        arm, sample, original = task
        qid = sample["question_id"]
        path = folder / "judge_repeat" / arm / f"{qid}.json"
        if path.exists():
            cached = read(path)
            assert cached["answer_sha256"] == digest(original["answer"])
            return cached
        prompt = judge_prompt(sample, original["answer"])
        response = bedrock.converse(
            cfg["judge_model"], prompt, "judge_repeat", max_tokens=16,
            question_id=qid, arm=arm,
        )
        label = response.lower().strip().rstrip(".")
        if label not in ("yes", "no"):
            raise ValueError(f"Invalid repeated judge label: {response!r}")
        row = {
            "arm": arm, "question_id": qid, "observed_at": time.time(),
            "answer_sha256": digest(original["answer"]),
            "prompt_sha256": digest(prompt), "original_correct": original["correct"],
            "repeat_correct": label == "yes", "judge_response": response,
        }
        write(path, row)
        return row

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        rows = list(pool.map(score, tasks))
    changed = [r for r in rows if r["original_correct"] != r["repeat_correct"]]
    summary = {
        "n": len(rows), "agreements": len(rows) - len(changed),
        "agreement_rate": 1 - len(changed) / len(rows), "changed": changed,
        "by_arm": {
            arm: {
                "n": sum(r["arm"] == arm for r in rows),
                "original_correct": sum(r["original_correct"] for r in rows if r["arm"] == arm),
                "repeat_correct": sum(r["repeat_correct"] for r in rows if r["arm"] == arm),
            } for arm in ARMS
        },
        "protocol": "Same frozen answers, official prompt, Sonnet 4.5 and temperature=0. "
                    "All answers rejudged because spot checks found questionable labels. "
                    "Diagnostic only; primary scores unchanged. Agreement is not judge accuracy.",
    }
    write(folder / "judge_repeat_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="longmemeval_20261116")
    audit(parser.parse_args().run)
