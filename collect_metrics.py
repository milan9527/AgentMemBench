"""Capture AWS service metrics for owned benchmark strategies without double counting."""
import argparse
from datetime import datetime, timezone

import boto3

from benchmark import ROOT, read, write


def collect(run):
    folder = ROOT / "runs" / run
    manifest = read(ROOT / "runs/aws_resources.json")
    control = boto3.client("bedrock-agentcore-control", region_name=manifest["region"])
    cw = boto3.client("cloudwatch", region_name=manifest["region"])
    quotas = boto3.client("service-quotas", region_name=manifest["region"])
    resources, metrics = {}, {}
    start = datetime.strptime(manifest["run_prefix"].removeprefix("amlbench_"), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc)
    for name, item in manifest["resources"].items():
        resources[name] = control.get_memory(memoryId=item["id"])["memory"]
        metrics[name] = {}
        for metric_name in ("TokenCount", "Throttles"):
            definitions = []
            for page in cw.get_paginator("list_metrics").paginate(
                Namespace="AWS/Bedrock-AgentCore", MetricName=metric_name,
                Dimensions=[{"Name": "StrategyId", "Value": item["strategy_id"]}],
            ):
                definitions.extend(page["Metrics"])
            if not definitions:
                metrics[name][metric_name] = {"unavailable": True}
                continue
            # AWS also publishes a resource-only aggregate; never add it to strategy totals.
            definition = max(definitions, key=lambda d: len(d["Dimensions"]))
            data = cw.get_metric_statistics(
                Namespace=definition["Namespace"], MetricName=metric_name,
                Dimensions=definition["Dimensions"], StartTime=start, EndTime=end,
                Period=60, Statistics=["Sum"],
            )
            points = sorted(data["Datapoints"], key=lambda x: x["Timestamp"])
            metrics[name][metric_name] = {"definition": definition, "points": points,
                                          "sum": sum(x["Sum"] for x in points)}
    quota_rows = []
    for page in quotas.get_paginator("list_service_quotas").paginate(ServiceCode="bedrock-agentcore"):
        quota_rows += [q for q in page["Quotas"] if any(w in q["QuotaName"].lower()
                                                      for w in ("memor", "extract", "event"))]
    write(folder / "aws_configuration.json", {"observed_at": end, "resources": resources, "quotas": quota_rows})
    write(folder / "cloudwatch_metrics.json", {
        "start": start, "end": end, "metrics": metrics,
        "caveat": "CloudWatch may arrive late; aggregate processing metrics are not model input/output or invoice usage. Includes diagnostics and retries.",
    })
    print({name: {k: v.get("sum") for k, v in series.items()} for name, series in metrics.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="longmemeval_20261116")
    collect(parser.parse_args().run)
