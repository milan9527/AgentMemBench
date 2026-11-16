# LongMemEval-S 首轮结果存档

本目录保存 2026-11-16 的固定报告与汇总，不包含完整运行输入、原始候选和逐条模型日志。

| 文件 | 内容 |
|---|---|
| [REPORT.md](REPORT.md) | 14 题 × 9 方案的质量、延迟、费用及局限 |
| [summary.json](summary.json) | 主评测汇总 |
| [latency_summary.json](latency_summary.json) | 写入结束后的预热检索延迟 |
| [judge_repeat_summary.json](judge_repeat_summary.json) | 全部 126 个固定回答的复判一致性 |
| [timestamp_semantics_audit.json](timestamp_semantics_audit.json) | 后端时间字段语义差异 |
| [config.json](config.json) | 本轮非敏感运行配置 |
| [检索诊断报告](retrieval_diagnostics_v1/REPORT.md) | 12 题的事实覆盖、排序和预算损失 |
| [指标 CSV](retrieval_diagnostics_v1/metrics.csv) | 各 K 值及 4K 上下文指标 |
| [诊断汇总](retrieval_diagnostics_v1/summary.json) | 完整结构化诊断汇总 |
| [诊断完整性检查](retrieval_diagnostics_v1/integrity.json) | 本地原始诊断完成时的校验记录 |
| [复核协议](retrieval_diagnostics_v1/review_protocol.json) | 自动单记录复核的选择规则 |

报告正文保持原样，提及的 `results/`、`retrieval/`、`snapshots/`、`labels/`、`inputs.json`、`gold_evidence.json`、`manual_labels.json`、`protocol.json`、`analysis.json` 和其他未列出的审计文件仍在原始本地运行归档中。完整性检查对应该本地归档，不能被解读为仅凭本仓库即可复查全部逐条模型证据。

主报告中的资源清理状态是报告生成时的记录，不是当前云端资源状态。
检索诊断采用预先定义的事实粒度、自动标签及明确记录的 Codex 裁决；小样本和已知时间歧义限制了结论的适用范围。
