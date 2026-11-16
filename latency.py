"""Randomized, warm retrieval-only timings after memory ingestion has finished."""
import argparse
import random
import time

import numpy as np

from benchmark import (ROOT, STRATEGIES, AgentCoreBackend, Bedrock, Journal, Mem0Backend,
                       actor_id, rag_search, read, write)


def run(name, repeats):
    folder = ROOT / "runs" / name
    cfg, samples = read(folder / "config.json"), read(folder / "samples.json")
    journal = Journal(folder / "usage.jsonl")
    mem0, ac, br = Mem0Backend(cfg, folder, journal), AgentCoreBackend(cfg, folder, journal), Bedrock(cfg, journal)
    arms = ("rag", "mem0", *STRATEGIES)
    rng = random.Random(cfg["seed"])
    rows = []

    def measure(sample, arm, repeat):
        actor, qid = actor_id(folder, sample), sample["question_id"]
        if arm == "mem0":
            records, elapsed = mem0.search(actor, sample["question"], qid)
        elif arm == "rag":
            records, elapsed = rag_search(br, folder, qid, sample["question"], cfg["top_k"])
        else:
            records, elapsed = ac.search(arm, actor, sample["question"], qid)
        return {"question_id": qid, "arm": arm, "repeat": repeat, "seconds": elapsed,
                "records": len(records), "observed_at": time.time()}

    warmups = [measure(samples[0], arm, -1) for arm in arms]
    write(folder / "latency_warmups.json", warmups)
    for repeat in range(repeats):
        jobs = [(s, a) for s in samples for a in arms]
        rng.shuffle(jobs)
        for sample, arm in jobs:
            rows.append(measure(sample, arm, repeat))
            write(folder / "latency_raw.json", rows)
        print(f"Latency repetition {repeat + 1}/{repeats} finished", flush=True)
    summary = {}
    for arm in arms:
        values = [r["seconds"] for r in rows if r["arm"] == arm]
        summary[arm] = {"n": len(values), "p50_seconds": float(np.median(values)),
                        "p95_seconds": float(np.percentile(values, 95)), "warmups_excluded": True}
    write(folder / "latency_summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="longmemeval_20261116")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(args.run, args.repeats)
