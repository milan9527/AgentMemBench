# AgentMemBench — Mem0 OSS / AgentCore Memory 实测

这是内部 LongMemEval-S 分层评测，不是 agentmemoryleaderboard.ai 的正式提交，也不能与该榜单分数直接比较。

2026-11-16 已完成 14 题 × 9 方案的实测，完整历史共约 144 万 tokens。
结果、延迟、费用拆分和逐题疑点见 [完整报告](reports/longmemeval_20261116/REPORT.md)。
首轮 LLM 判分为 Mem0 12/14、AgentCore S+P+U 融合 13/14；一题差距来自存在时间歧义的样本，
不足以判定产品优劣。全部 126 个固定回答复判一致，但仍发现两例判分疑点，未事后改分。

本轮固定 `mem0ai==2.2.1`、Qdrant 1.20.1、Bedrock Haiku 4.5 回答/抽取、Sonnet 4.5 评分、Titan Text Embeddings V2。AWS 区域为 `us-east-1`，使用当前 AWS credential chain，不在配置中保存密钥。

## 方案与口径

| arm | 方案 |
|---|---|
| none | 不检索历史，测量模型先验和拒答基线 |
| rag | 完整原始历史分块 + Titan 向量检索 |
| mem0 | 自建 Mem0 OSS，默认事实抽取和混合检索，关闭 rerank |
| semantic / preference / summary / episodic | AgentCore 四种内置策略，分别创建独立 Memory |
| spu | Semantic + Preference + Summary 的检索结果按排名交错合并 |
| spue | 在 spu 基础上加入 Episodic |

每题使用独立 actor/user_id，完整历史按原始日期顺序导入，不注入问题、标准答案或证据标签。每条消息保留角色，并统一添加原始会话日期；约 3K token / 最多 20 条消息一块，不跨会话。原始会话 ID 重复时用日期区分。Mem0 OSS 不支持 `timestamp` 参数，因此原始日期通过消息正文和 metadata 传入。

所有方案候选上限 100、回答模型相同、最终记忆正文上限 4,096 `cl100k_base` tokens（包含分隔符）。预算不是完整模型输入大小，Bedrock 实际计费 token 另记。组合策略每个分支取最多 100，合并后共用 4K 预算，因此候选数量可超过单策略；这是额外检索开销，报告单列。组合复用已返回的分支候选，延迟为各分支串行调用耗时之和，不是实际并行组合服务延迟。

返回的原始时间信息也计入这 4K 预算：Mem0 使用 `metadata.source_date`，AgentCore 使用原生
`timestamp`，统一标为 `Memory source timestamp`。
本轮发现这些字段的语义并不完全一致：Summary 的部分记录返回 2026 年处理时间，
另一些返回 2023 年历史时间；Episodic 也有少量处理时间。该统一标签不能视为已归一化的来源日期，
也不能当作事实发生日期。这是当前配置的限制，详见运行目录的 `timestamp_semantics_audit.json`；
下一轮应明确区分事件时间、记录时间和事实时间。RAG 的日期已经包含在每条原始消息中。

Mem0 使用上游默认抽取提示词，AgentCore 使用内置默认策略。因此测的是可部署的系统配置，无法控制托管服务的内部抽取模型。此版本 Mem0 OSS 和 Mem0 托管 Platform 不是同一产品；这些结果不能代表 Mem0 Platform。

## 复现

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
docker run -d --name mem0-agentcore-benchmark-qdrant \
  -p 127.0.0.1:6335:6333 \
  -v "$PWD/.cache/qdrant:/qdrant/storage" \
  qdrant/qdrant@sha256:d18c8e5061be3eaea2f53fbee1f0bf0382dc59cbb9b7112e41ffcc937ec14220
.venv/bin/python resources.py provision
.venv/bin/python benchmark.py prepare --run my_run
.venv/bin/python benchmark.py ingest --run my_run
.venv/bin/python benchmark.py repair --run my_run
# 等待后台抽取；readiness 记录快照和失败任务，必要时再次 repair。
.venv/bin/python resources.py readiness --run my_run
.venv/bin/python benchmark.py evaluate --run my_run
.venv/bin/python benchmark.py report --run my_run
.venv/bin/python analyze.py --run my_run
```

`finish_run.py --run my_run` 可在导入进程已启动后等待 checkpoint，并依次完成评分、失败抽取重试、
跨时间快照稳定性检查、预热后的检索延迟测量和报告。它不会自动创建资源或清理资源。

`audit_judges.py --run my_run` 对全部固定回答统一复判，仅输出诊断文件，不覆盖主结果；
之后重新运行 `analyze.py` 可把判分一致性和额外模型费用加入报告。

`data/source_manifest.json` 固定 Hugging Face 数据版本和 SHA256。下载地址：
`https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json`。
下载后应先校验 SHA256，再运行 prepare。评分提示词来自 [LongMemEval 官方脚本](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)，本地副本及 SHA256 在 `vendor/`；评分模型替换为 Sonnet，故也不是原版评测配置。

`--limit 1` 可以试跑第一题；`--offset 1` 跳过第一题。已成功的写入分块和答题支持续跑。不要为同一个 actor 同时启动两个写入进程；Mem0 在写入成功但 checkpoint 尚未保存时异常退出，不能保证完全幂等。AgentCore 使用稳定 clientToken。

`config.json` 的 `samples_per_category` 决定每个类别的数量；当前为 2，即 14 题。共有 user / assistant / preference / multi-session / temporal / update / abstention 七层，拒答题由 `_abs` ID 单独划分。样本抽取只依赖 ID、类别和 seed，不依赖答案或实际成绩。

## 异步抽取与错误

AgentCore `CreateEvent` 成功只表示接受事件。当前账号长期抽取配额为 150K tokens/min，客户端用 100K tokens/min 的输入估计值平滑限速；托管提示词、合并和重试产生的额外用量不可见，限速后仍须检查失败任务。`repair` 调用 `StartMemoryExtractionJob` 重试 `LTM_RATE_EXCEEDED`，保留原始事件。其他失败须调查，不作为模型答错处理。

评测要求 Semantic / Preference / Summary 最后一次写入或重试后至少等待 300 秒，Episodic 至少等待 1,200 秒，且没有可见 FAILED jobs。本轮观察到短会话的 Episodic 记录在约 15 分钟空闲后集中生成，因此不能套用 5 分钟等待。`readiness` 保存跨时间快照，用于检查记录是否仍在变化。记录稳定与失败任务为空都不是服务端“全部完成”的权威证明，尤其 reflection 可能更晚产生；报告必须保留这个限制。不要把 CreateEvent 延迟当成记忆可检索延迟。

首次连通性试跑曾快速提交一个用户的事件，触发后台限速；所有重试和额外调用保留在运行日志。质量评分前修复这些任务。修复期间的系统开销单列，不能与稳定工作负载的延迟直接比较。

## 输出与清理

`runs/<run>/results/` 保存逐题回答、标准答案、评分、实际上下文；`retrieval/` 保存原始检索候选；`snapshots/` 和 `readiness/` 保存 AgentCore 记录与异步状态；`usage.jsonl` 保存模型 token 和调用时间；`ingestion/` 保存写入 checkpoint。数据、日志和本地数据库不纳入 git。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python resources.py cleanup
docker stop mem0-agentcore-benchmark-qdrant
```

cleanup 只删除 `runs/aws_resources.json` 中本脚本创建并带有基准标签的四个 Memory；清理前保存快照。删除 Memory 后已存档的答题/报告仍可复查，重新在线检索需要重新 provision、建立新的 run 并导入历史。Qdrant 数据卷保留本地。
已有同名且已停止的 Qdrant 容器可用 `docker start mem0-agentcore-benchmark-qdrant` 启动，无需重复创建。

正式选型还需要扩大题量、多次重复、业务数据和工具任务；本轮 QA 不能证明 Episodic 的工具执行成功率或经验迁移效果。运维、隔离、删除能力和生产 SLO 应另测。

## 检索诊断

已完成的[检索诊断报告](reports/longmemeval_20261116/retrieval_diagnostics_v1/REPORT.md)与
[指标 CSV](reports/longmemeval_20261116/retrieval_diagnostics_v1/metrics.csv)补充 Recall@K、Precision@K、Hit@K、MRR、pooled nDCG，以及实际 4K 回答上下文的事实覆盖率。
12 道可回答题中，Mem0 与 AgentCore S+P+U 的 4K 标注事实覆盖率均为 100%；这不等于回答正确率或生产环境召回率。事实标签由 Codex 整理并用 Sonnet 自动核验，尚未独立人工校准。

基于已经完成并保留候选、上下文的本地运行执行：

```bash
.venv/bin/python diagnose_retrieval.py all --run my_run
```

该命令会调用 Bedrock 进行标签判定并产生额外费用；不会重新导入记忆或生成回答。
当前 `retrieval_diagnostics_gold.py` 的事实标注覆盖本轮 12 道可回答题；更换样本集时需要补充对应标注。
本轮报告还包含独立存档的 Codex 标签裁决与原消息精确匹配修正，重新自动判定不保证完全复现这些分数。

## 仓库中的实验材料

[结果目录说明](reports/longmemeval_20261116/README.md)列出已发布材料与本地审计材料的边界。
仓库包含代码、依赖锁、上游来源及许可证、数据下载版本与校验值、主报告和诊断汇总。
完整数据集、原始会话、检索候选、模型请求日志、数据库、AWS 资源清单与凭据保留在运行环境中。
报告中提及的 `runs/` 路径及未发布审计文件是原始运行归档路径，需通过本地运行产生。
本仓库中的报告是固定实验快照；新的运行结果仍写入被 Git 忽略的 `runs/` 目录。
