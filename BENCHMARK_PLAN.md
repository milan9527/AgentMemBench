# 自部署 Mem0 与 AgentCore Memory 对比评测设计

状态：评测设计，尚未运行实验或创建 AWS 资源。
文档核对日期：2026-11-16。

## 目标与范围

回答三个问题：

1. 在相同 agent、任务和读取预算下，AgentCore 托管 Memory 与自部署 Mem0，谁的任务效果、时效和成本更合适？
2. AgentCore 的 Semantic、User Preference、Summary、Episodic 单独使用及组合使用时，各自在哪类任务上有效？
3. 内置配置不足时，覆盖模型或提示词、自管提取流程，分别能带来多少收益？

用户已明确自建方案采用 Mem0。本评测的对象为 Mem0 OSS 自部署实例与 AgentCore Memory 托管服务。
Mem0 的部署方式、版本、模型及数据库尚待选定；本设计不预设已有实例。
AgentCore overrides 和 self-managed 仅作为后续解释收益的可选实验，不作为第一轮必需项。

## 已核实的产品边界

| 类型 | 提取和整理 | 存储与检索 | 比较对象 |
| --- | --- | --- | --- |
| AgentCore built-in | AWS 托管默认流程 | AgentCore | 开箱效果 |
| AgentCore built-in with overrides | AWS 托管流程，指定模型和提示词 | AgentCore | 模型、提示词定制收益 |
| AgentCore self-managed strategy | 自建提取和整理流程 | AgentCore | 自定义处理流程收益 |
| Mem0 OSS 自部署 | Mem0 流程，自选模型与配置 | 自选数据库及部署资源 | 整套产品效果与成本 |

Overrides 保留内置输出 schema；要改变 schema 和完整处理流程，使用 self-managed。
托管内置流程中的模型未必能与自建模型完全对齐，因此跨系统比较应解释为整套方案比较，不能直接归因为某个算法优劣。

## 实验一：四种策略的覆盖能力及组合收益

同一份事件历史分别写入独立实验资源，使用同一 agent 和读端模型。

| 实验组 | 配置 | 目的 |
| --- | --- | --- |
| B0 | 无长期记忆，保留统一的当前会话窗口 | 衡量长期记忆净收益 |
| B1 | 原始历史分块 RAG | 简单检索基线 |
| M0 | Mem0 OSS，自部署、固定版本和完整配置 | 与托管服务直接比较 |
| S | Semantic | 用户事实、信息更新 |
| P | User Preference | 偏好、风格、选择约束 |
| U | Summary | 会话上下文、决策和任务进展 |
| E | Episodic，读取 episodes 与 reflections | 工具使用、失败经验、流程复用 |
| SPU | Semantic + User Preference + Summary | 对话助手组合 |
| SPUE | 四种策略组合 | 同时覆盖对话和执行经验 |

第一轮共九组：两个基础对照、Mem0、四种 AgentCore 单策略、两个 AgentCore 组合。
先比较单策略和两个组合。需要解释组合收益时，再增加：

- SPUE 去掉一个策略，共四组，测各策略在组合中的边际贡献。
- Episodic 仅读取 episodes，与读取 episodes + reflections 对比。
- Summary 只读 chunks、只读 full-session summary 的读取方式对比。

后两类是读取消融；如生成流程仍产生全部记录，不应声称节省了生成费用。
移除策略后的效果差同时包含内容互补和上下文竞争，不能将其解释为独立、可相加的贡献。

## 实验二：AgentCore 托管与 Mem0 自部署

比较实验一中预先选定的 AgentCore 配置、M0、B0、B1。
AgentCore 配置在独立开发集上确定，不根据最终测试集挑选最好配置。
上下文容量允许时，增加完整历史输入，作为参考基线；它不是理论上限。

主实验固定：

- agent 框架、回答模型及版本、提示词、工具、当前会话窗口。
- 输入事件、角色、会话边界、顺序及可见时间信息。
- 记忆查询生成方式、最大查询轮数和总记忆上下文 token 预算。
- 多策略候选融合、去重和裁剪规则；规则在开发集上固定。
- 评分标准、评分模型和测试样本。

各系统保留自身提取与检索实现。内部实现差异是被测产品的一部分。
如还需评估每套产品充分调优后的表现，可另设“各自调优”实验，保持相同外部成本或延迟约束，并完整记录配置。
两种实验的结果分别报告。

### Mem0 的实验配置

M0 使用 Mem0 OSS 的正常记忆提取与搜索流程，而非仅把原始文本直接存入向量库。
第一轮选定一个数据库后固定，不同时把数据库选型引入主对比。
可采用固定配置的 Qdrant 或 pgvector；选型本身不表示预判哪种效果更好。

必须保存以下配置：

- `mem0ai` 版本、源码 commit 或镜像 digest，及依赖锁文件。
- library 嵌入调用或自托管 server；两种调用边界不能混在同一个延迟统计中。
- 提取 LLM、embedding 模型、提示词、数据库版本、索引及硬件。
- NLP/实体提取组件、BM25 支持是否实际启用；依赖缺失导致的降级须在试跑发现。
- 候选数、阈值、reranker 及其模型；不要依赖会随版本变化的默认值。
- 写入粒度、并发度、用户/session 映射和历史保留方式。

Mem0 官方 README 当前明确说明，其公布的高分来自托管 Platform，含 OSS 未提供的优化。
这些分数不能作为本次自部署 Mem0 的实验结果。
当前 OSS 迁移文档描述 ADD-only 提取和混合检索，并已移除旧外部 graph-store 集成。
因此不默认添加“Mem0 + Neo4j”组；如实际使用旧版本，需单独标记版本与配置并实测。

信息更新按最终行为评分：新事实是否被正确使用、历史时点问题是否仍可回答。
不要求 Mem0 和 AgentCore 必须通过同样的覆盖或删除操作实现更新。

如果 M0 在特定业务切片上不足，再增加 M1：使用相同基础设施与模型、在独立开发集上定制 Mem0 提取指令或检索参数。
M1 与 AgentCore overrides 都是可选调优实验，和默认配置结果分开。

### 统一接入边界

评测运行器组织相同的原始事件，分别交给两个适配器：

```text
同一份历史事件
  ├─ Mem0Adapter ────── Mem0 正常提取与存储
  └─ AgentCoreAdapter ─ CreateEvent 与配置策略

同一个问题
  → 固定的查询生成规则
  → 各后端检索
  → 统一的总 token 裁剪及上下文封装
  → 同一回答模型 / agent
  → 同一评分器
```

适配器负责写入、处理进度观测、检索、隔离和使用量记录；它不持有答案与证据标签。
评测运行器负责最终回答，不能让 Mem0 示例 chatbot 和另一个不同 agent 分别作答。
先采用一次逻辑检索、无额外查询改写的读端基线；后续如开放多次检索，对所有组开放相同轮数。
AgentCore 多策略可能需要多个物理 API 调用，须计入延迟与成本。

Mem0 按 user_id 隔离同一人的跨会话记忆；如写入了 run_id，跨会话查询不能误用仅当前 run 的过滤条件。
同一人的 user_id 在一个实验组内稳定，在不同实验运行中隔离；session 信息保留，不用新 user_id 代替每个新 session。
AgentCore 则使用相应的 actor 和 namespace 范围。
工具消息若需适配为后端可接受的格式，保留原始角色及出处并记录转换，不能把工具结果伪装成用户陈述。

阈值数值不强行一致：Mem0 混合排序分数与 AgentCore 相似度分数的含义可能不同。
固定各组已声明的阈值与候选策略，以相同总记忆 token 预算评价实际送入回答模型的证据。
对同一 memory budget（例如 2K、4K、8K）绘制质量曲线；预算点是实验设置，不是服务限制。

### 托管与自部署的成本、延迟

- Mem0 包括提取 LLM、embedding、数据库、计算、备份，以及需要的检索模型费用。
- AgentCore 包括事件、长期记忆、检索等实际使用项目；如使用 overrides，加入单独计费的模型调用。
- 两边都加入回答模型费用；公共评分器费用另列为评测开销。
- 按相同历史量、查询量、保留期估算月成本，并同时报告空闲成本和新增用户的边际成本。
- 维护与运维投入单列为估算，不混入实测 API 费用。
- 从同一 agent 所在位置测端到端查询延迟，部署在可比地域并记录调用边界、并发与冷热状态。
- Mem0 写入调用耗时与 AgentCore 事件受理耗时不是相同指标；统一补测“提交事件到新记忆可检索”的时间。

## 实验三：定制 AgentCore 的收益（可选）

按业务相关策略逐项测试，不同时修改所有变量：

1. 默认内置策略。
2. Overrides，指定模型，使用官方基础提示词。
3. 与第 2 组使用相同模型，增加领域提示词。
4. Self-managed，自定义提取和整理，继续使用 AgentCore 存储检索。

第 2 与第 3 组可用于研究领域提示词的收益。
第 1 与第 2 组还可能包含底层模型差异，不应称为纯提示词实验。
如要进一步研究存储检索差异，把同一套自定义生成的记录分别导入 AgentCore 和自建存储，以统一查询与预算比较；嵌入、索引和排序差异仍需记录。

## 数据与评分

| 测试切片 | 例子 | 主要指标 | 数据来源 |
| --- | --- | --- | --- |
| 用户事实 | 订单号、项目约束、当前配置 | 问答正确率、事实覆盖 | LongMemEval + 业务样本 |
| 偏好使用 | 输出语言、差旅偏好、编码风格 | 约束遵守率、正确工具参数 | LongMemEval 偏好题 + 工具任务 |
| 会话延续 | 之前决定了什么、当前进度如何 | 状态正确率、遗漏率 | 业务多会话轨迹 |
| 时间和更新 | 从旧地址改成新地址 | 当前事实正确率、旧信息误用率 | LongMemEval + 业务样本 |
| 不可回答 | 历史未提供的信息 | 正确拒答率、无依据回答率 | LongMemEval |
| 执行经验 | 工具失败后的替代步骤、重复工作流 | 任务成功率、重复错误率、调用次数 | 自有轨迹、参考 MemoryArena |
| 记忆驱动行动 | 根据早期规则选择目标、日期或收件人 | 工具、目标、参数与内容正确性 | 自有模拟任务、参考 DolphinBench |

主公开集使用 LongMemEval 官方 cleaned 版本，保存 revision/hash。
保留其原始分类并报告官方完整集分数；额外提供“用户事实、助手信息、偏好”等切片，解释策略覆盖范围。
不能为了适配 Semantic 删除助手信息题后仍声称是完整 LongMemEval 分数。

LoCoMo 可作为补充；它是两个人物的对话，不能不加说明地把其中一个人物映射成 assistant，从而改变策略可提取的信息。
若增加角色适配版本，单独命名并报告映射方式，和原版结果分开。

业务集至少包含偏好变更、信息撤销、无关干扰、时间关系和已完成的工具执行过程。
调优使用独立开发集；最终测试集冻结并记录版本。
标准答案、证据标签、任务评分条件均不得进入记忆输入或查询上下文。

## 执行协议

1. 每个实验组使用独立资源或可验证隔离的命名空间；隔离 run、actor 和 session。
2. 按事件顺序回放历史，保留 USER、ASSISTANT、TOOL 等角色。不得把全部内容改成 USER 以提高某策略分数。
3. 所有组都只接收查询发生前的历史。若服务时间戳有限制，采用统一、保留相对顺序和间隔的映射，并记录该转换。
4. 分别运行两种时间协议：
   - 处理完成后的质量：依据可用处理日志、状态和记录快照确认进度，设置固定超时。
   - 固定延迟的质量：在预先设定的写入后时间点查询，画出效果随等待时间变化的曲线。
5. 不使用标准答案判断“是否完成”，也不等待到预期事实出现才开始计分。没有可靠完成信号时，明确标注固定等待或稳定性启发式的局限。
6. 回答测试使用新会话和隔离的工具状态。历史问答测试不把测试问题或答案写回公共记忆。
7. 连续经验学习测试另设协议：允许任务间写入，但所有组收到相同外部任务；记录各自行动和反馈，防止与静态回放分数混用。
8. 记录原始检索候选、进入模型的记忆、回答、工具调用、等待时间、费用和错误。

### AgentCore 特有的注意事项

- Semantic 的默认提取指令以用户消息为事实来源，助手消息仅辅助理解。
- Semantic 和 User Preference 只处理 USER、ASSISTANT 角色；工具经验应通过 Episodic 等合适策略评估。
- Summary 是会话级别的：跨会话查询需要覆盖历史 session 的命名空间，不能只查当前 session。
- Summary 当前有 chunks 与 FULL_SESSION 两类记录，须固定读取策略，避免重复内容占用预算。
- Episodic 在检测到 episode 完成后生成记录；反思还需要跨 episode 处理。数据必须包括完成信号、结果及足够的历史 episode。
- Episodic 官方建议提供 TOOL 结果；其检索索引使用 episode intent 和 reflection use case。
- Reflection 范围显式限制到实验规定的 actor；除非专门测共享知识，不混用其他 actor 的经验。
- 多策略的总 token 预算必须一致。每种策略各取相同 top-k，不等于整体预算公平。
- CreateEvent 请求成功不等于长期记忆处理结束；不把异步未完成一概当成质量错误。
- IngestData 不持久化短期事件，且不触发 self-managed；涉及 self-managed 的主对照优先统一使用 CreateEvent 回放。

## 指标与统计

主要结论以问答正确率或任务成功率为准，并逐能力切片报告。

诊断指标：

- 提取：有依据的有效事实覆盖、无依据记录、错误合并、旧事实误用。
- 检索：在给定总 token 预算下，必要证据的覆盖率。
- 使用：已检索到必要证据但答案或行动仍错误的比例。
- 时效：新事实首次可用时间、固定等待时限下的成功率、处理超时率。
- 性能：查询与端到端 p50/p95 延迟、工具调用量、输入输出 token。
- 成本：历史写入、模型提取和整理、存储、检索、回答、后台处理；按共同保留期及查询量摊销。

托管与自建记录的粒度不同，不直接以“记忆记录 ID 的 Recall@k”比较质量。
将公开证据或业务标注拆成必要事实单元，评价这些事实是否被检索内容支持；对主观评分抽样人工复核。
工具名称、收件人、日期等可确定字段优先采用确定性规则；语义内容采用统一评分器。

对相同样本做配对比较。按独立历史、actor 或完整任务链聚类 bootstrap，避免把同一历史里的问题当作相互独立。
区分：

- 固定记忆，多次查询：衡量读端随机性。
- 重新构建记忆后再查询：衡量完整流程随机性。

正式重复次数和样本量根据试跑中的方差、最小有业务意义的差异和预算确定。
不预先声称少量样本能证明小幅提升。

## 分阶段运行

1. 接口与口径试跑：分层抽取约 30–50 道问题、少量完整工具轨迹，验证输入映射、隔离、异步处理和费用记录。
2. 第一轮比较：B0、B1、M0、四种单策略、SPU、SPUE，共九组；公开问答与工具经验分别展示。
3. 解释结果：仅对有价值的组合做去掉单策略实验，并测试必要的 overrides 或 self-managed。
4. 正式报告：冻结数据、配置和代码，跑完整选定测试集；输出质量—成本、质量—延迟及能力切片对照。

试跑用于工程验证和估算，调优使用独立开发集。
实际调用前须确定 AWS region、Mem0 版本与部署配置、模型、目标工作负载和费用预算。

## 报告格式

| 配置 | 事实 | 偏好 | 状态延续 | 信息更新 | 经验任务成功率 | p95 查询延迟 | 记忆可用延迟 | 每用户月成本 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 无长期记忆 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 不适用 | 待测 |
| 原始历史 RAG | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| AgentCore 各策略及组合 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Mem0 OSS（固定版本及配置） | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

月成本需注明输入历史量、每月查询次数、保存时长及冷启动/稳定运行假设。
综合分只在业务权重预先确定后给出，不用一个总分掩盖不同策略的用途。

## Agent Memory Leaderboard（AML）适用性

核对日期：2026-11-16。结论：适合作为 Mem0 OSS 与 AgentCore 的统一质量评测协议参考，正式平台可用性和完整本地复现另行确认。
AML 是多个数据集、任务和统一运行协议组成的评测平台，不是单一数据集。

### 与本项目的对应关系

- Textual：参与方负责 Add/Search，平台固定 Answer/Eval。适合比较事实、偏好、时间、更新等质量，并包含 Streaming 增量写入与查询。
- Coding：平台提供 coding agent 和 verifier；同一组 150 个基础任务分别在 relevant/noisy 条件下执行，共 300 个任务条件单元。适合补充工程经验复用与抗干扰测试。
- AgentCore 策略消融：内部可用相同任务和评分器比较 S/P/U/E/SPU/SPUE。正式平台对近似版本、重复提交和配额有限制，不能假定九组都可直接申请代跑。
- 成本、写入新鲜度、生产并发和 p95：仍须在本项目运行器独立采集；公开 Overall 不能替代这些指标。

### 必须处理的接入差异

1. AML 的 Add 成功表示该批记忆已处理且可检索。AgentCore CreateEvent 仅受理事件，适配器必须等待实际处理完成；不能立即返回成功。网站协议允许经审核绑定状态查询端点后使用 HTTP 202。
2. 每次 Add 可以只覆盖一段会话。Episodic 可能要等后续片段才能检测到 episode 完成，因此逐块 Add 的完成要求可能与 episode 生成时机冲突。先用完整/未完成 episode 的 smoke 验证处理语义；无法在协议内满足时，Episodic 改用内部、允许跨块完成的运行器单独评测，不修改事件语义或伪造结束信号。
3. AML Search 以 user_id 为检索范围，不提供 session_id 过滤。AgentCore Summary 适配器必须检索该 actor 的历史 session，Mem0 也不能只过滤当前 run。
4. Textual 正式查询采用 top_k=100，统一回答窗口为 128,000 tokens，总输入预算为 117,760 tokens，超限按返回顺序裁剪。该口径可做官方质量比较，但还应增加内部 2K/4K/8K 记忆预算曲线，以反映业务上下文与费用约束。
5. 开源方法榜要求固定 embedding（text-embedding-v4）和 LLM 相关组件（gpt-4o-mini），商业榜没有同样限制。不能将受限 Mem0 与不受限 AgentCore 的两张榜单分数称为同条件算法比较。内部评测保留两套生产配置，固定读端与评分器，并披露写端差异；正式参测类别由平台规则与审核确定。

### 可复现范围和运行规模

公开仓库包含若干 Answer/Eval pipeline（包括 LongMemEval-S、LoCoMo-refined、BEAM 等），但不发布完整原始评测包、私有问题、金标、私有标注和内部运行设施。
下载仓库不等于可以本地完整复现官方 Full 与 Overall。使用原始公开数据集自跑时，必须标明是内部子集和所采用的评分版本，不能标为官方 AML 分数。

官方 README 声明 Cycle 2 Textual 为 6,000+ 样本、32 个来源、约 3 亿 tokens；Coding 约 18 亿 tokens。
这些是官方声明的赛道数据规模，不等于每家厂商的实际计费 token；对多组配置应先估算回放、提取及重复运行成本。

当前网站显示 Cycle 2 材料截止为 2026-10-31，正式评测截止为 2026-11-04，结果预计 2026-11-19 发布。当前不能假定新申请可立即运行正式评测。
先使用其协议和公开 pipeline 完成内部试跑；官方代跑与公榜作为后续选项。

## 一手资料

- [AgentCore 策略分类](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-strategies.html)
- [Semantic](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/semantic-memory-strategy.html)
- [User Preference](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/user-preference-memory-strategy.html)
- [Summary](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/summary-strategy.html)
- [Episodic](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/episodic-memory-strategy.html)
- [Built-in with overrides](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-custom-strategy.html)
- [Self-managed](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-self-managed-strategies.html)
- [检索接口](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/long-term-retrieve-records.html)
- [异步 IngestData](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/long-term-ingest-data.html)
- [Memory 可观测性](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-observability.html)
- [AgentCore Evaluations](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations.html)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
- [MemoryArena](https://memoryarena.github.io/)
- [DolphinBench](https://github.com/mem0ai/dolphinbench)
- [Mem0 README：OSS 与 Platform 及评测口径](https://github.com/mem0ai/mem0)
- [Mem0 OSS 概览](https://docs.mem0.ai/open-source/overview)
- [Mem0 OSS 新算法迁移说明](https://docs.mem0.ai/migration/oss-v2-to-v3)
- [Mem0 Python Memory 实现](https://github.com/mem0ai/mem0/blob/main/mem0/memory/main.py)
- [AML 官方仓库与方法说明](https://github.com/AML-memory/agent-memory-leaderboard)
- [AML 用户文档](https://agentmemoryleaderboard.ai/docs)
- [AML Add/Search 接口协议](https://agentmemoryleaderboard.ai/api-guide)
- [AML 赛事与当前时间安排](https://agentmemoryleaderboard.ai/competition/)
