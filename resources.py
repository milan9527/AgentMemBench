"""Manage only resources listed in this benchmark's ownership manifest."""
import argparse
import time
from datetime import datetime, timezone

import boto3

from benchmark import ROOT, STRATEGIES, AgentCoreBackend, Mem0Backend, Journal, actor_id, read, write

PATH = ROOT / "runs/aws_resources.json"


def provision():
    cfg = read(ROOT / "config.json")
    if PATH.exists() and not read(PATH).get("cleaned_up"):
        raise RuntimeError("An ownership manifest already exists; reuse its resources or clean up first")
    client = boto3.client("bedrock-agentcore-control", region_name=cfg["region"])
    prefix = "amlbench_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest = {"region": cfg["region"], "run_prefix": prefix, "resources": {}}
    keys = {
        "semantic": "semanticMemoryStrategy", "preference": "userPreferenceMemoryStrategy",
        "summary": "summaryMemoryStrategy", "episodic": "episodicMemoryStrategy",
    }
    for kind in STRATEGIES:
        strategy = {"name": f"Benchmark_{kind}", "namespaces": ["/bench/{actorId}/"]}
        if kind == "summary":
            strategy["namespaces"] = ["/bench/{actorId}/sessions/{sessionId}/"]
        if kind == "episodic":
            strategy["namespaces"] = ["/bench/{actorId}/episodes/"]
            strategy["reflectionConfiguration"] = {"namespaces": ["/bench/{actorId}/"]}
        memory = client.create_memory(
            name=f"{prefix}_{kind}", description="Isolated Mem0 versus AgentCore benchmark",
            eventExpiryDuration=7, memoryStrategies=[{keys[kind]: strategy}],
            tags={"Purpose": "mem0-agentcore-benchmark", "ManagedBy": "memory-benchmark"},
        )["memory"]
        manifest["resources"][kind] = {"id": memory["id"], "strategy": strategy, "status": memory["status"]}
        write(PATH, manifest)
        print(kind, memory["id"], memory["status"], flush=True)
    while True:
        statuses = []
        for kind, item in manifest["resources"].items():
            m = client.get_memory(memoryId=item["id"])["memory"]
            item.update(status=m["status"], strategy_id=m["strategies"][0]["strategyId"])
            statuses.append(m["status"])
        write(PATH, manifest)
        print("Resource statuses:", statuses, flush=True)
        if all(s == "ACTIVE" for s in statuses):
            return
        if any(s == "FAILED" for s in statuses):
            raise RuntimeError("Memory provisioning failed; inspect ownership manifest")
        time.sleep(15)


def readiness(run):
    folder = ROOT / "runs" / run
    cfg = read(folder / "config.json")
    backend = AgentCoreBackend(cfg, folder, Journal(folder / "usage.jsonl"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for sample in read(folder / "samples.json"):
        qid = sample["question_id"]
        for strategy in STRATEGIES:
            path = folder / "ingestion" / strategy / f"{qid}.json"
            if not path.exists() or "finished_at" not in read(path):
                continue
            actor = actor_id(folder, sample)
            records = backend.snapshot(strategy, actor)
            jobs = backend.failed_jobs(strategy, actor)["jobs"]
            row = {"question_id": qid, "strategy": strategy, "observed_at": time.time(),
                   "records": records, "failed_jobs": jobs}
            write(folder / "readiness" / stamp / strategy / f"{qid}.json", row)
            rows.append({"question_id": qid, "strategy": strategy, "records": len(records),
                         "failures": len(jobs)})
    write(folder / "readiness" / stamp / "summary.json", rows)
    print(rows, flush=True)


def snapshot_mem0(run):
    folder = ROOT / "runs" / run
    backend = Mem0Backend(read(folder / "config.json"), folder, Journal(folder / "usage.jsonl"))
    for sample in read(folder / "samples.json"):
        actor = actor_id(folder, sample)
        records = backend.memory.get_all(filters={"user_id": actor}, top_k=10000)["results"]
        if len(records) == 10000:
            raise RuntimeError("Snapshot may be truncated; use a paginated export")
        if any(r.get("user_id") != actor for r in records):
            raise RuntimeError("Snapshot actor isolation check failed")
        write(folder / "snapshots/mem0" / f"{sample['question_id']}.json",
              {"observed_at": time.time(), "records": records})
    print("Mem0 snapshots saved", flush=True)


def cleanup():
    manifest = read(PATH)
    client = boto3.client("bedrock-agentcore-control", region_name=manifest["region"])
    for strategy, item in manifest["resources"].items():
        if item.get("deleted_at"):
            continue
        m = client.get_memory(memoryId=item["id"])["memory"]
        tags = client.list_tags_for_resource(resourceArn=m["arn"])["tags"]
        if tags.get("Purpose") != "mem0-agentcore-benchmark" or not m["name"].startswith(manifest["run_prefix"]):
            raise RuntimeError("Ownership verification failed; refusing deletion")
        client.delete_memory(memoryId=item["id"])
        item["deleted_at"] = datetime.now(timezone.utc).isoformat()
        item["status"] = "DELETING"
        write(PATH, manifest)
        print("Deleted benchmark resource:", strategy, item["id"], flush=True)
    manifest["cleaned_up"] = True
    write(PATH, manifest)
    for _ in range(20):
        pending = 0
        for item in manifest["resources"].values():
            if item["status"] == "DELETED":
                continue
            try:
                client.get_memory(memoryId=item["id"])
                pending += 1
            except client.exceptions.ResourceNotFoundException:
                item["status"] = "DELETED"
                write(PATH, manifest)
        if not pending:
            print("All owned cloud memories are deleted", flush=True)
            return
        time.sleep(3)
    raise RuntimeError("Deletion requested, but some resources are still deleting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["provision", "readiness", "snapshot_mem0", "cleanup"])
    parser.add_argument("--run", default="longmemeval_20261116")
    args = parser.parse_args()
    globals()[args.command](args.run) if args.command in ("readiness", "snapshot_mem0") else globals()[args.command]()
