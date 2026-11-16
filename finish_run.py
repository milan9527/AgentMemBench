"""Resume post-ingestion phases without hiding incomplete extraction or API failures."""
import argparse
import subprocess
import sys
import time

from benchmark import ROOT, STRATEGIES, AgentCoreBackend, Journal, actor_id, digest, read, write
from resources import readiness


def finish(run):
    folder = ROOT / "runs" / run
    cfg, samples = read(folder / "config.json"), read(folder / "samples.json")
    ac = AgentCoreBackend(cfg, folder, Journal(folder / "usage.jsonl"))

    def command(script, *args):
        subprocess.run([sys.executable, str(ROOT / script), *args, "--run", run], check=True)

    def wait_ingested(backends):
        while True:
            remaining = []
            for s in samples:
                for backend in backends:
                    p = folder / "ingestion" / backend / f"{s['question_id']}.json"
                    if not p.exists() or "finished_at" not in read(p):
                        remaining.append((s["question_id"], backend))
            if not remaining:
                return
            print(f"Waiting for ingestion: {len(remaining)} actor/backend jobs remain", flush=True)
            time.sleep(30)

    wait_ingested(["mem0"])
    if not all((folder / "rag" / f"{s['question_id']}.npz").exists() for s in samples):
        raise RuntimeError("RAG indexing is incomplete; finish ingestion before continuing")
    command("benchmark.py", "evaluate", "--arms", "none", "rag")
    command("resources.py", "snapshot_mem0")
    command("benchmark.py", "evaluate", "--arms", "mem0")
    wait_ingested(STRATEGIES)
    # Do not run repair concurrently with bulk ingestion: both consume the same quota.
    for round_no in range(10):
        pending_wait = 0
        for s in samples:
            for backend in STRATEGIES:
                state = read(folder / "ingestion" / backend / f"{s['question_id']}.json")
                latest = max(state["finished_at"], state.get("last_repair_at", 0))
                minimum_wait = cfg.get("agentcore_episodic_wait_seconds", 1200) if backend == "episodic" else cfg["agentcore_wait_seconds"]
                pending_wait = max(pending_wait, minimum_wait - (time.time() - latest))
        while pending_wait > 0:
            print(f"Waiting for asynchronous extraction: {pending_wait:.0f}s minimum remaining", flush=True)
            time.sleep(min(30, pending_wait))
            pending_wait -= 30
        failures = []
        for s in samples:
            for strategy in STRATEGIES:
                jobs = ac.failed_jobs(strategy, actor_id(folder, s))["jobs"]
                if jobs:
                    failures.append({"question_id": s["question_id"], "strategy": strategy, "jobs": jobs})
        write(folder / "repair_rounds" / f"{round_no}.json", failures)
        print(f"Repair round {round_no}: {sum(len(x['jobs']) for x in failures)} failed jobs", flush=True)
        if not failures:
            break
        command("benchmark.py", "repair")
    else:
        raise RuntimeError("Extraction remains incomplete after ten repair rounds; inspect failures")

    readiness(run)
    # Compare record sets, including content, across two observations. This is only a heuristic.
    def record_signatures():
        signatures = {}
        for s in samples:
            for strategy in STRATEGIES:
                records = ac.snapshot(strategy, actor_id(folder, s))
                signature = sorted((r["memoryRecordId"], r.get("content", {})) for r in records)
                signatures[f"{s['question_id']}:{strategy}"] = digest(signature)
        return signatures

    previous = record_signatures()
    for observation in range(10):
        time.sleep(30)
        time.sleep(30)
        current = record_signatures()
        changes = [key for key in current if current[key] != previous[key]]
        write(folder / "stability" / f"{observation}.json",
              {"observed_at": time.time(), "signatures": current, "changed_since_previous": changes})
        print(f"Readiness observation {observation}: {len(changes)} changing actor/strategy record sets", flush=True)
        if not changes:
            break
        previous = current
    else:
        raise RuntimeError("Memory still changing; refusing to claim a stable evaluation snapshot")
    readiness(run)
    command("benchmark.py", "evaluate", "--arms", *STRATEGIES, "spu", "spue")
    command("latency.py")
    command("benchmark.py", "report")
    command("analyze.py")
    print("FINISHED_ALL_EVALUATIONS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="longmemeval_20261116")
    finish(parser.parse_args().run)
