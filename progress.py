"""Read local checkpoints; does not send any cloud requests."""
import argparse
from datetime import datetime, timezone

from benchmark import ARMS, ROOT, STRATEGIES, read

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="longmemeval_20261116")
args = parser.parse_args()
folder = ROOT / "runs" / args.run
n = read(folder / "manifest.json").get("history_chunks")
if n is None:
    n = read(folder / "protocol_manifest.json")["history_chunks"]
counts = {}
for backend in ("mem0", *STRATEGIES):
    states = [read(p) for p in (folder / "ingestion" / backend).glob("*.json")]
    counts[backend] = {
        "users_finished": sum("finished_at" in s for s in states),
        "chunks": sum(len(s["completed"]) for s in states),
    }
print(datetime.now(timezone.utc).isoformat())
print(counts)
print("AgentCore import:", round(sum(counts[k]["chunks"] for k in STRATEGIES) / (n * 4) * 100, 1), "%")
print("Scored questions:", {arm: len(list((folder / "results" / arm).glob("*.json"))) for arm in ARMS})
log = folder / "finish.log"
if log.exists() and log.read_text().splitlines():
    print("Controller:", log.read_text().splitlines()[-1])
print("Logs with error markers:", [
    p.name for p in folder.glob("*.log") if any(x in p.read_text() for x in ("ERROR", "Traceback"))
])
