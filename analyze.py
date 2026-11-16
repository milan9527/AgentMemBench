"""Produce auditable pilot statistics; never treat missing jobs as wrong answers."""
import argparse
from collections import defaultdict
import math
from pathlib import Path

import numpy as np

from benchmark import ARMS, CATEGORIES, ROOT, STRATEGIES, read, write

NAMES = {
    "none": "无记忆", "rag": "原始历史 RAG", "mem0": "Mem0 OSS",
    "semantic": "AgentCore Semantic", "preference": "AgentCore Preference",
    "summary": "AgentCore Summary", "episodic": "AgentCore Episodic",
    "spu": "AgentCore S+P+U 融合", "spue": "AgentCore S+P+U+E 融合",
}


def wilson(successes, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p, z2 = successes / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [max(0, center - half), min(1, center + half)]


def paired(rows, baseline, seed=20261116):
    ids = sorted(set(rows) & set(baseline))
    if not ids:
        return {}
    diffs = np.array([int(rows[i]["correct"]) - int(baseline[i]["correct"]) for i in ids])
    wins, losses = int(np.sum(diffs > 0)), int(np.sum(diffs < 0))
    discordant = wins + losses
    p = min(1, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) /
            2 ** discordant) if discordant else 1
    rng = np.random.default_rng(seed)
    strata = [diffs[[rows[i]["category"] == c for i in ids]] for c in CATEGORIES]
    strata = [x for x in strata if len(x)]
    # Paired resampling within each selected stratum. Pilot CI, no population calibration.
    boot = np.concatenate([rng.choice(x, (10000, len(x))) for x in strata], axis=1).mean(axis=1)
    return {"n": len(ids), "wins": wins, "losses": losses, "ties": len(ids) - discordant,
            "accuracy_delta": float(diffs.mean()), "stratified_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
            "mcnemar_exact_p_unadjusted": p}


def analyze(run):
    folder = ROOT / "runs" / run
    cfg, manifest = read(folder / "config.json"), read(folder / "manifest.json")
    samples = read(folder / "samples.json")
    ids = {s["question_id"] for s in samples}
    all_rows = {
        arm: {p.stem: read(p) for p in (folder / "results" / arm).glob("*.json") if p.stem in ids}
        for arm in ARMS
    }
    complete = all(len(rows) == len(samples) and all(x["status"] == "ok" for x in rows.values())
                   for rows in all_rows.values())
    summary = {"complete": complete, "n": len(samples), "arms": {}, "usage": {}, "ingestion": {}}
    for arm, by_id in all_rows.items():
        rows = list(by_id.values())
        ok = [r for r in rows if r.get("status") == "ok"]
        correct = sum(r["correct"] for r in ok)
        summary["arms"][arm] = {
            "n": len(ok), "correct": correct, "accuracy": correct / len(ok) if ok else None,
            "wilson_95ci_descriptive": wilson(correct, len(ok)),
            "by_category": {c: {"correct": sum(r["correct"] for r in ok if r["category"] == c),
                                "n": sum(r["category"] == c for r in ok)} for c in CATEGORIES},
            "paired_vs_mem0": paired(by_id, all_rows["mem0"]),
            "retrieval_p50": float(np.median([r["retrieval_seconds"] for r in ok])) if ok else None,
            "retrieval_p95": float(np.percentile([r["retrieval_seconds"] for r in ok], 95)) if ok else None,
            "mean_context_tokens": float(np.mean([r["context_tokens"] for r in ok])) if ok else None,
            "empty_retrievals": sum(r["retrieved_count"] == 0 for r in ok),
            "answerable_correct": sum(r["correct"] for r in ok if r["category"] != "abstention"),
            "answerable_n": sum(r["category"] != "abstention" for r in ok),
        }
    # Prices from AWS Pricing GetProducts, us-east-1 geographic inference, captured 2026-11-16.
    # These are public-list estimates, NOT Cost Explorer or a reconciled invoice.
    rates = {"haiku": (1.1, 5.5), "sonnet": (3.3, 16.5), "titan": (.02, 0)}
    usage = [__import__("json").loads(line) for line in (folder / "usage.jsonl").read_text().splitlines()]
    groups = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_model_usd": 0.0})
    for item in usage:
        group = groups[item["stage"]]
        group["calls"] += 1
        tokens = item.get("usage", {})
        i, o = tokens.get("inputTokens") or 0, tokens.get("outputTokens") or 0
        group["input_tokens"] += i
        group["output_tokens"] += o
        model = item.get("model", "")
        if model:
            key = next((k for k in rates if k in model), None)
            if key is None:
                raise ValueError(f"Missing public price for {model}")
            group["estimated_model_usd"] += (i * rates[key][0] + o * rates[key][1]) / 1e6
    summary["usage"] = dict(groups)
    summary["public_price_estimated_model_usd"] = sum(g["estimated_model_usd"] for g in groups.values())
    summary["agentcore_retrieve_estimated_usd"] = groups["agentcore_search"]["calls"] * .0005
    for backend in ("mem0", *STRATEGIES):
        states = [read(p) for p in (folder / "ingestion" / backend).glob("*.json") if p.stem in ids]
        times = [s["finished_at"] - s["started_at"] for s in states if "finished_at" in s]
        summary["ingestion"][backend] = {
            "finished_users": len(times), "accepted_chunks": sum(len(s["completed"]) for s in states),
            "per_user_wall_seconds_median": float(np.median(times)) if times else None,
            "note": "Mem0 synchronous; AgentCore accepts events asynchronously; pacing/queueing included",
        }
        snapshots = [read(p) for p in (folder / "snapshots" / backend).glob("*.json") if p.stem in ids]
        summary["ingestion"][backend]["snapshotted_records"] = sum(len(s["records"]) for s in snapshots)
        summary["ingestion"][backend]["snapshotted_users"] = len(snapshots)
    summary["native_cost_components"] = {
        backend: {
            "accepted_events": summary["ingestion"][backend]["accepted_chunks"],
            "ingestion_usd_lower_bound_decimal_units": summary["ingestion"][backend]["accepted_chunks"] * 12000 / 1e9,
            "ingestion_usd_upper_bound_decimal_units": summary["ingestion"][backend]["accepted_chunks"] * 64000 / 1e9,
            "snapshot_records": summary["ingestion"][backend]["snapshotted_records"],
            "constant_stock_monthly_ltm_usd": summary["ingestion"][backend]["snapshotted_records"] * .00075,
            "snapshot_complete": summary["ingestion"][backend]["snapshotted_users"] == len(samples),
        }
        for backend in STRATEGIES
    }
    if (folder / "latency_summary.json").exists():
        summary["controlled_retrieval_latency"] = read(folder / "latency_summary.json")
    if (folder / "judge_repeat_summary.json").exists():
        summary["judge_repeat"] = read(folder / "judge_repeat_summary.json")
    if (folder / "cleanup_status.json").exists():
        summary["cleanup"] = read(folder / "cleanup_status.json")
    write(folder / "analysis.json", summary)

    lines = [
        "# Mem0 OSS 与 AgentCore Memory：首轮实测",
        "",
        f"状态：{'全部完成' if complete else '运行中，禁止作为最终比较'}。"
        f"英文 LongMemEval-S cleaned 分层 {len(samples)} 题；7 类各 {cfg['samples_per_category']} 题；"
        f"完整历史 {manifest['history_tokens']:,} tokens。不是官方 AML 成绩，也不是完整 LongMemEval 成绩。",
        "",
        "统一回答模型 Haiku 4.5、评分模型 Sonnet 4.5；记忆正文预算 4,096 tokens。"
        "Mem0 OSS 2.2.1 + 自建 Qdrant 1.20.1；AgentCore 使用默认内置策略。配置和复现步骤见根目录 README.md。",
        "",
        "## 答题质量",
        "",
        "| 方案 | 首轮 LLM 判对 / 完成 | 准确率 | 描述性 Wilson 95% 区间 | 平均记忆 tokens | 空检索题数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, s in summary["arms"].items():
        if not s["n"]:
            continue
        low, high = s["wilson_95ci_descriptive"]
        lines.append(f"| {NAMES[arm]} | {s['correct']}/{s['n']} | {s['accuracy']:.1%} | "
                     f"{low:.1%}–{high:.1%} | {s['mean_context_tokens']:.0f} | {s['empty_retrievals']} |")
    lines += [
        "",
        "分层样本被人为等权，整体准确率是本轮七类等权的 pilot 分数，不能估计真实业务流量准确率。"
        "Wilson 区间仅作小样本不确定性的描述；两题一层不足以可靠估计层内分布。"
        "逐题配对胜负、分层 bootstrap 和未做多重比较校正的 McNemar p 值保存在 analysis.json；不能据此宣布稳定排名。",
        "",
        "## 分类型结果",
        "",
        "| 方案 | 用户事实 | 助手信息 | 偏好 | 跨会话 | 时间推理 | 信息更新 | 拒答 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, s in summary["arms"].items():
        lines.append("| " + NAMES[arm] + " | " + " | ".join(
            f"{s['by_category'][c]['correct']}/{s['by_category'][c]['n']}" for c in CATEGORIES) + " |")
    lines += ["", "## 延迟", "", "质量评测时的检索耗时如下。包含网络和 SDK，但不包含回答/评分；"
              "组合项为实际分支耗时之和。小样本及并行导入负载会影响这些值，不能当作生产 SLO。",
              "", "| 方案 | p50 秒 | p95 秒 |", "|---|---:|---:|"]
    for arm, s in summary["arms"].items():
        if s["retrieval_p50"] is not None:
            lines.append(f"| {NAMES[arm]} | {s['retrieval_p50']:.3f} | {s['retrieval_p95']:.3f} |")
    if "controlled_retrieval_latency" in summary:
        lines += ["", "另在写入结束后按随机顺序重复测量检索，预热后每题 3 次；"
                  "详细结果见 latency_summary.json。该结果更适合比较本机环境下的检索开销。",
                  "", "| 方案 | 调用数 | p50 秒 | p95 秒 |", "|---|---:|---:|---:|"]
        for arm, s in summary["controlled_retrieval_latency"].items():
            lines.append(f"| {NAMES[arm]} | {s['n']} | {s['p50_seconds']:.3f} | {s['p95_seconds']:.3f} |")
    lines += [
        "", "## 用量与费用", "",
        "下表 token 来自真实 Bedrock 响应，费用按 2026-11-16 AWS Pricing API 的公开单价估算，"
        "不是账单实付值。US 地理推理 Haiku 输入/输出 $1.10/$5.50 每百万 token，"
        "Sonnet $3.30/$16.50，Titan 输入 $0.02。原始价格快照在 runs/bedrock_price_list.json。",
        "", "| 阶段 | 调用数 | 输入 tokens | 输出 tokens | 模型费用估算 USD |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, g in groups.items():
        if g["input_tokens"] or g["output_tokens"]:
            lines.append(f"| {stage} | {g['calls']} | {g['input_tokens']:,} | "
                         f"{g['output_tokens']:,} | {g['estimated_model_usd']:.4f} |")
    lines += [
        "", f"全部可观测模型费用估算 **${summary['public_price_estimated_model_usd']:.4f}**，"
        f"包含 Mem0 抽取、两个 embedding 用途、全部方案回答和公共评分。"
        f"AgentCore Retrieve 调用估算另计 **${summary['agentcore_retrieve_estimated_usd']:.4f}**。",
        "",
        "AgentCore SDK 不返回内置抽取模型及输入/输出 token；CloudWatch 可提供聚合 TokenCount，"
        "它的统计口径与模型账单不同，不能按 Mem0 的 token 单价推算。"
        "当前公开价格为 STM 导入 $1/GB（每事件最少 12KB、最多 64KB），STM 存储 $0.10/GB-month，"
        "内置 LTM 存储 $0.75/千记录/月，检索 $0.50/千请求。"
        "模型费用合计不包含这些存储/导入项目、EC2/Qdrant、磁盘、网络、运维、税费或失败请求潜在费用，"
        "因此不能用上述小计宣称某方案总成本更低。"
        "本轮为每位用户建立完整历史后仅回答一道质量题，生产环境的多次复用会改变成本摊销。"
        "价格来源：https://aws.amazon.com/bedrock/agentcore/pricing/ 及 runs/agentcore_price_list.json。",
        "",
        "## 完整性与限制",
        "",
        "- 首个样本的快速导入触发 AgentCore LTM_RATE_EXCEEDED；其后平滑限速并重放失败抽取任务。"
        "重试调用保存在 usage.jsonl。拒绝给仍存在可见 FAILED jobs 的样本评分。",
        "- AgentCore 评分前要求最近写入/重试后至少 300 秒，Episodic 至少 1,200 秒，并保存抽取记录快照。"
        "本轮观察到短会话的 Episodic 记录在约 15 分钟后集中生成，因此早期空检索不用于最终评分。"
        "没有权威的后台完成信号；观察到稳定记录也不保证所有 reflection 已完成。",
        "- 所有完整历史按时间顺序输入，标准答案仅用于评分；没有按成绩挑选样本。"
        "Mem0 原始日期在正文/metadata 中，AgentCore 另使用原始 eventTimestamp。",
        "- 回答上下文中的 Memory source timestamp 直接使用后端返回字段，语义未归一化。"
        "这个标签不能等同于事实或事件发生时间，可能影响时间推理。"
        "原始日期还通过统一消息前缀输入；本轮字段审计见 timestamp_semantics_audit.json。",
        "- S+P+U / S+P+U+E 是跨独立策略资源的检索组合，复用相同单策略记录；"
        "不代表单个 Memory 上所有策略共同运行时的性能或成本。",
        "- 默认事实策略偏向用户信息；助手信息、摘要和偏好的用途不同，应结合分类结果选策略。",
        "- QA 数据没有真实工具执行评价，不能据此判断 Episodic 的经验迁移、动作复用或任务成功率。",
        "- 主结果采用单次 reader + 首轮 LLM judge；后续复判仅作诊断。"
        "不含跨随机种子稳定性或人工复核一致率。"
        "后续应扩大样本、使用业务轨迹并补充工具任务。",
        "",
        "逐题证据：results/<arm>/<question_id>.json；检索：retrieval/；异步状态：snapshots/ 和 readiness/；"
        "模型调用：usage.jsonl；完整结构化统计：analysis.json。",
    ]
    if all(c["snapshot_complete"] for c in summary["native_cost_components"].values()):
        lines += [
            "", "## AgentCore 用量对应的公开价格", "",
            "下表导入费用范围按每事件最低 12KB、最高 64KB 的公开计费规则计算，采用十进制单位作近似。"
            "仅含 14 题成功导入的事件，不含两个隔离诊断事件。"
            "月存储列是假设记录数量不变并保留整月的价格，不是本次运行账单；实际删除前按小时计费。"
            "查询、STM 存储、公共回答模型等需另加，S+P+U 等融合方案按其使用的分支计费。",
            "", "| 单策略 | 成功导入事件 | 导入费范围 USD | 快照记录数 | 保持该规模一月的 LTM 存储 USD |",
            "|---|---:|---:|---:|---:|",
        ]
        for backend, c in summary["native_cost_components"].items():
            lines.append(f"| {NAMES[backend]} | {c['accepted_events']} | "
                         f"{c['ingestion_usd_lower_bound_decimal_units']:.4f}–"
                         f"{c['ingestion_usd_upper_bound_decimal_units']:.4f} | "
                         f"{c['snapshot_records']} | {c['constant_stock_monthly_ltm_usd']:.4f} |")
        lines += ["", "Mem0 另需向量库与运行主机；本次使用现有主机上的 Qdrant，未将整台主机费用分摊给它。"
                  "上述拆分可用于建立成本模型，但不能把一次抽取费与一个月存储费直接比较。"]
    if (folder / "data_quality_notes.json").exists():
        lines += [
            "", "## 证据复查发现的时间歧义", "",
            "以下题目仍按原始标签评分，没有事后剔除或改分。证据原文在 data_quality_notes.json。",
            "",
            "- `8550ddae`：5 月 30 日提问“上周末试过什么鸡尾酒”，标注证据是 5 月 22 日会话中的"
            "“last weekend”。Lavender Gin Fizz 这一事实有支持，但相对时间参照发生了变化。"
            "Mem0 的回答已提到该配方，仍因时间条件而拒绝直接作答。",
            "- `37f165cf`：标准答案 856 有 440 + 416 页的数值支持；问题指定“一月和三月”，"
            "标注证据会话却在 5 月并表述为“just finished”，其中未见对应月份条件。"
            "Mem0 的回答已列出两个页数，但没有接受题目的月份前提。",
            "",
            "因此，答题准确率包含抽取、检索、时间理解、回答策略和数据标注共同造成的影响。"
            "这两例不能简单归类为“没有记住”。扩大评测前应审计时间题的证据和标签；"
            "本轮原始分数保留可复核性，不使用事后修正分数宣传效果。",
        ]
    if (folder / "timestamp_semantics_audit.json").exists():
        lines += [
            "", "另有后端时间字段差异：本轮 Summary 候选中 646/1352 条返回 2026 年处理时间，"
            "其余为 2023 年历史时间；Episodic 也有 4/1279 条返回 2026 年。"
            "来源字段标签不代表时间语义一致。后续扩大评测前应区分事件时间、记录时间和事实时间。",
        ]
    if (folder / "review_notes.json").exists():
        lines += [
            "", "## 自动评分与融合检索的复查", "",
            "以下为 Codex 的证据复查，不是独立人工标注；所有首轮分数保持原样，详见 review_notes.json。",
            "",
            "- `8550ddae / episodic`：回答拒答且没有给出标准答案 lavender gin fizz，首轮 judge 却判 yes。",
            "- `2133c1b5_abs / spue`：回答指出住在 Harajuku 而非 Shinjuku，与拒答解释相符，"
            "首轮 judge 却判 no；附加的日期推理本身也有矛盾。",
            "- `1a8a66a6`：S+P+U 正确回答两份杂志；加入 Episodic 后，"
            "Architectural Digest 的记录仍在候选中，但因较长记录占用共同的 4K 预算而未进入回答上下文，"
            "S+P+U+E 只回答一份。这是当前融合和截断方式下的失败案例，不证明 Episodic 本身无用。",
        ]
    if "judge_repeat" in summary:
        repeat = summary["judge_repeat"]
        lines += [
            "", "发现判分疑点后，对全部已固定答案使用相同模型、提示词和 temperature=0 统一复判，"
            "不重新生成回答、不覆盖首轮结果。",
            f"两次判分一致 {repeat['agreements']}/{repeat['n']}（{repeat['agreement_rate']:.1%}）。"
            "这衡量重复判分的一致性，不是 judge 的正确率；复判不能代替独立人工校准。",
            "", "| 方案 | 首轮判对 | 复判判对 |", "|---|---:|---:|",
        ]
        for arm, values in repeat["by_arm"].items():
            lines.append(f"| {NAMES[arm]} | {values['original_correct']}/{values['n']} | "
                         f"{values['repeat_correct']}/{values['n']} |")
        lines += ["", "逐项差异在 judge_repeat_summary.json。后续选型应先校准判分规则与时间题标签，"
                  "避免用一题差距作产品优劣结论。"]
    if complete and run == "longmemeval_20261116":
        disagreements = [
            qid for qid in ids
            if all_rows["spu"][qid]["correct"] != all_rows["mem0"][qid]["correct"]
        ]
        lines += [
            "", "## 本轮可支持的选型观察", "",
            f"- Mem0 {summary['arms']['mem0']['correct']}/{len(samples)}，"
            f"AgentCore S+P+U 融合 {summary['arms']['spu']['correct']}/{len(samples)}；"
            f"逐题分歧：{', '.join(f'`{q}`' for q in sorted(disagreements)) or '无'}。"
            "本次唯一分歧是上述时间歧义题，不能据此判定产品优劣。",
            "- AgentCore Semantic 在本轮两道助手信息题均未答对，Summary 两道均答对；"
            "组合体现了不同策略的信息覆盖差异。Preference 的总体分较低，但两道偏好题均答对，"
            "总体 QA 分数不等于其专项目标表现。",
            "- 固定上下文预算下，融合策略需要控制各分支占用和相关性。"
            "本轮更适合支持下一轮比较 Mem0 与 Semantic+Preference+Summary，"
            "并为 Episodic 单独增加具有工具执行和经验迁移的任务。",
        ]
    if "cleanup" in summary:
        cleanup = summary["cleanup"]
        cloud_cleanup = (
            "本次创建的四个 AgentCore Memory 已确认删除。"
            if cleanup["all_owned_memories_deleted"]
            else "本次创建的四个 AgentCore Memory 均已提交删除；最近检查云端仍在异步删除中，"
                 "尚未确认全部消失。可再次运行 resources.py cleanup 验证状态。"
        )
        lines += [
            "", "## 资源清理", "",
            cloud_cleanup + f"本次 Qdrant 容器已停止：{cleanup['qdrant_stopped']}。"
            "本地数据库、完整输入、回答、检索证据和费用快照保留，详见 cleanup_status.json。"
            "未修改其他已有 Memory 或停止宿主机。",
        ]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(folder / "REPORT.md")
    print("Completed:", complete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="longmemeval_20261116")
    analyze(parser.parse_args().run)
