<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/brand/banner-dark.svg">
  <img src="docs/brand/banner-light.svg" alt="Tenet — 面向智能体的双时间轴信念记忆：无需图数据库的时间正确性" width="820">
</picture>

<p>
  <a href="paper/tenet.pdf"><b>📄 论文</b></a> ·
  <a href="paper/extended_abstract.md"><b>扩展摘要</b></a> ·
  <a href="docs/BENCHMARK.md"><b>基准测试</b></a> ·
  <a href="docs/COMPARISON.md"><b>对比 Mem0 / Zep / Letta</b></a> ·
  <a href="src/tenet/mcp_server.py"><b>MCP 服务器</b></a> ·
  <a href="scripts/demo_agent.py"><b>演示</b></a>
</p>

[![tests](https://github.com/Nas01010101/tenet/actions/workflows/test.yml/badge.svg)](https://github.com/Nas01010101/tenet/actions/workflows/test.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/tenet.pdf)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg?logo=python&logoColor=white)](#快速开始)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-yellow.svg?logo=pypi&logoColor=white)](pyproject.toml)
[![Qwen Cloud](https://img.shields.io/badge/built%20on-Qwen%20Cloud-6a5acd.svg)](https://qwencloud-hackathon.devpost.com)
[![MCP](https://img.shields.io/badge/MCP-native-000000.svg)](src/tenet/mcp_server.py)
[![stars](https://img.shields.io/github/stars/Nas01010101/tenet?style=flat&color=8b7cf8)](https://github.com/Nas01010101/tenet/stargazers)

*读取记忆不应该消耗一次 LLM 调用。*

**[English README](README.md)** · 本翻译由 Anthropic 的 Claude（Fable 5）完成，并使用开源机器翻译系统
[Argos Translate](https://github.com/argosopentech/argos-translate)（LibreTranslate 所用引擎）
回译交叉校验；所有数字与技术术语均与英文原版逐一核对。以英文版为准。

```bash
pip install tenet-memory   # 尚未发布到 PyPI —— 在此之前请从源码安装（见下文）
```
```python
from tenet import Tenet

mem = Tenet()
mem.ingest("I live in Boston")              # 需要 LLM 密钥（将原始消息蒸馏为事实）
mem.ingest("I moved to Seattle")            # 触发取代（supersession）—— Boston 保留在历史中
mem.recall("where do I live?")              # → [Seattle]（当前信念，无 LLM 调用）
mem.recall("where do I live?", as_of=t0)    # → [Boston]（时间回溯，无 LLM 调用）
mem.navigate("where do I live and work?")   # → 自适应多跳召回，无 LLM 调用
```
`recall` / `stats` / `doubts` / 时间回溯（`recall(as_of=...)`）/ `navigate` 全部 **不调用 LLM** ——
只用嵌入向量 + 余弦相似度 + 闭式数学公式，毫秒级延迟；配合 `EMBED_PROVIDER=local`
时完全不需要任何 API 密钥。`ingest`（以及聊天智能体）需要可用的
`DASHSCOPE_API_KEY`（或 `LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY`），因为把自由文本
转化为原子事实是唯一需要模型判断的环节——这条界线的确切位置见下文
[60 秒零密钥演示](#快速开始)。

</div>

---

## Tenet 对比 Zep · Mem0 · Letta

2026 年的智能体记忆领域按用途分化：**Mem0** 做单用户个性化，**Zep/Graphiti**
处理随时间变化的事实，**Letta** 做自我管理的长程智能体。Tenet 瞄准的是 Zep 的场景——
*事实变化时的时间正确性*——但移除了它的准入成本。

| | **Tenet** | Zep / Graphiti | Mem0 | Letta |
|---|---|---|---|---|
| 随时间变化的事实 | ✅ 双时间轴取代机制 | ✅ 双时间轴图 | ❌ 仅创建时间戳 | 智能体自管理 |
| **运行所需基础设施** | **`pip install` —— 仅 sqlite + numpy** | 图数据库（Neo4j / FalkorDB） | 向量数据库 | 智能体服务器 + Postgres |
| 读取路径成本 | **无 LLM 调用** | 无 LLM 调用 | 无 LLM 调用 | 每次操作一次 LLM 调用 |
| **能直接读懂它记住了什么？** | ✅ **纯文本信念状态**（`get_all()`） | ❌ 图节点 | ❌ 不透明向量 | ❌ 状态块 |
| 即插即用 API | ✅ **兼容 Mem0**（`add`/`search`/`get_all`/`delete`） | 图 API | `add`/`search`/… | 完整运行时 |
| 时间回溯（`as_of`） | ✅ | ✅ | ❌ | ❌ |

**一句话总结：** *Zep 的时间正确性 + Mem0 的即插即用 API + 一份你能直接打开阅读的信念状态
——零基础设施。* 表中其他所有时间感知系统都需要常驻数据库服务；Tenet 只是一个库。
与向量记忆或图记忆不同，Tenet 存储的内容是 **人类可读的** ——
`subject::attribute → value`、当前值与被取代值分明——因此你可以精确审计智能体到底相信什么。

```python
mem = Tenet()
mem.add("I moved to Seattle", user_id="alex")     # Mem0 风格，即插即用
mem.search("where do I live?", user_id="alex")    # → [Seattle]（无 LLM 调用）
mem.get_all(user_id="alex")                        # → 可读的信念状态，而非不透明向量
```

完整的诚实对比矩阵与基准可比性注意事项：[`docs/COMPARISON.md`](docs/COMPARISON.md)。

## 结果一览

| 基准 | 指标 | Tenet | 对照 | 出处 |
|---|---|---:|---:|---|
| MemoryAgentBench FactConsolidation（ICLR 2026），单跳 | SubEM，6K–262K 汇总 | **86.5** [82.8, 89.5] | 已发表 mini 档 SOTA 78.0 · 朴素 RAG 47.8 | [`BENCHMARK.md` §6](docs/BENCHMARK.md#6-mab-factconsolidation--the-standardized-supersession-benchmark-scriptsbench_factconpy) |
| MAB Accurate-Retrieval | 官方指标平均 | **59.3**（全部已发表系统中第 2 名） | Mem0 32.6 · Zep 37.5 | [`BENCHMARK.md` §7](docs/BENCHMARK.md#7-mab-accurate-retrieval--the-second-mab-competency-scriptsbench_mab_arpy) |
| 知识翻新地平线（同一事实更新 2→12 次） | 当前值准确率 | **始终 100%** | 朴素 RAG 从 100% 塌到 50% | [`BENCHMARK.md` §3](docs/BENCHMARK.md#3-long-horizon-knowledge-churn--where-memory-structurally-wins-scriptsbench_horizonpy) |
| LongMemEval_S（n=100，`qwen3.7-plus` Qwen Cloud 阅读器） | 问答准确率 | **81.0%** | ≥ 同条件 RAG 79.0% · recall@10 **100%** · 上下文比全量少 **98.5%** | [`BENCHMARK.md` §1–2](docs/BENCHMARK.md#1-retrieval-recall--longmemeval_s-scriptslme_recallpy) |
| 本地 LoRA 蒸馏器（离线、零云端） | 键一致性（去污染评测） | **0.775** | 云端参考（`qwen3.7-plus`）0.707 | [`BENCHMARK.md` §10](docs/BENCHMARK.md#10-local-distiller-zero-cloud-verdict) |

诚实的弱项（多会话综合、多跳推理链）均如实报告、绝不隐藏——
完整表格与复现命令：[`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

## 读取记忆不应该消耗一次 LLM 调用

多数智能体记忆系统把 *读取* 路径架构在 LLM 之上——一次重排调用、一轮综合、
一个决定接下来取什么的智能体。**Tenet 的赌注恰好相反：**
`recall`、`doubts`、时间回溯（`recall(as_of=...)`）与自适应多跳 `navigate()` 全是纯向量相似度
+ 闭式数学，因此不产生任何 API 调用与推理延迟——真正需要判断力的环节
（把原始消息转化为原子化、带键的事实）只在 **写入时**（`ingest`）发生一次，而不是每次读取都发生。
取代机制本身——在事实变化时保持答案正确的核心机制——是确定性的双时间轴记账，同样没有模型参与。

这让读取路径*非常快*——并且在规模增长后依然快：

| 系统 | 读取/检索延迟 | 读取路径含 LLM | 运行所需基础设施 |
|---|---:|:---:|---|
| **Tenet** | **约 11 ms**（@100k 条事实，平坦） | **否** | 无 —— sqlite + numpy |
| Zep / Graphiti | 约 150–300 ms（图搜索） | 否 | 图数据库（Neo4j / FalkorDB） |
| Mem0 | 约 1.44 s p95（基础版） | 否 | 向量数据库 |
| Letta | 取决于模型（每次操作一次 LLM 调用） | 是 | 智能体服务器 + Postgres |

<sub>Tenet 的读取 = 嵌入 + 余弦 + 常驻矩阵上的闭式衰减 —— **从 1k 到 100k 条事实平坦地保持约 9–12 ms**
（[`docs/SCALE.md`](docs/SCALE.md)），比其自身未做常驻索引前的基线快约 100 倍。各系统的延迟*口径*不同
（均不含下游阅读器 LLM）；竞品数字取自各项目自己发表的检索延迟。重点不是跑分竞赛，而是：
这里的时间正确性 **既不需要图数据库，也不需要一次推理调用。**</sub>

LLM 智能体的记忆几乎总是 **对历史对话日志做检索**。对于要为*变化中的世界*建模的智能体，
这是错误的抽象：当同一事实在长交互中被反复更新——即 **知识翻新（knowledge churn）** ——
陈旧版本会挤占检索预算，智能体便用过期的值作答。**Tenet** 把记忆重构为
**自洽的信念状态**——关于用户的、当前有效且取代感知的事实集合——在检索式记忆崩溃之处保持正确。

<div align="center">

<img src="docs/brand/demo.gif" alt="Tenet 助手在事实变化时保持正确 —— 取代、时间回溯、遗忘" width="740">

<sub>真实录制会话：事实变化，信念状态将其取代，时间回溯召回之前为真的值——而读取路径从不调用 LLM。</sub>

</div>

## 没人测的失败模式

<div align="center">

![knowledge churn](docs/horizon.svg)

**当同一模板化事实被更新 2→12 次，RAG 式记忆从 100% 跌至 50%。Tenet 保持 100%。**

<sub>单属性翻新基元（`bench_horizon`）。在更难的*同义改写*、多属性翻新场景
（[ChurnBench §9](docs/BENCHMARK.md#9-churnbench--parametric-high-churn-stress-test-measured-2026-07-10)）下，
Tenet 默认开启的读取时一致性在 U=2/8/32 时分别达到 98/92/82 ——证伪过程与修复均完整报告。</sub>

</div>

## 为什么不一样

| | 检索式记忆（RAG） | **Tenet** |
|---|---|---|
| 抽象 | 对话轮的文档索引 | **双时间轴信念状态** |
| 事实变化时 | 两段相似的文本 | **被取代**（双时间轴，历史保留） |
| 陈旧证据 | 永远可被检索到 | **退役**（信念–证据一致性） |
| 写入策略 | 全部存储 | **惊奇度门控**（预测编码） |
| 遗忘 | 无（无限增长） | 显著性衰减清扫 |
| 事实漂移 | 未建模 | **陈旧度提示** —— 按属性学习 P(仍有效)，`tenet doubts` |
| 跨时间可查询 | 否 | **时间回溯**（`recall(as_of=t)`） |
| 多跳桥接 | 固定深度 *k* 或没有 | **自适应 `navigate()`** —— 仅当新证据越过相关性增益门限才加深跳数，无 LLM |
| 读取路径 | — | **无 LLM 调用** |

阅读 2 页论文：**[`paper/tenet.md`](paper/tenet.md)**。

## 快速开始

### 1. 60 秒，无需 API 密钥

```bash
pip install tenet-memory[local]             # bge-small 嵌入器，CPU —— 完全无网络调用
python examples/00_zero_key_demo.py         # 取代 + 时间回溯 + doubts，零 LLM 调用
```
端到端走完整条无 LLM 读取路径——召回、取代、时间回溯与学习型动态 `doubts`——
基于一份预生成的事实账本。它唯一无法展示的是 `ingest()` 将自由对话转化为这些事实的过程；
那是 Tenet 中唯一需要模型的调用（见下一步）。

### 2. 完整智能体（需要 API 密钥）

```bash
cp .env.example .env && chmod 600 .env      # 填入 DASHSCOPE_API_KEY（Qwen Cloud）
pip install -e ".[all]"                     # base + api/mcp/oss/local/cli/langgraph 扩展
python scripts/smoke_test.py                # 验证连通性
uvicorn tenet.api:app --host 0.0.0.0 --port 8000  # HTTP API，含 POST /chat
python -m tenet.mcp_server                   # 或 MCP 服务器（learn/recall/navigate/forget/stats）
```
仅 `pip install -e .` 只安装基础库（`openai`、`numpy`）——API 服务器与 MCP 服务器需要
`api`/`mcp` 扩展（上面的 `[all]` 已包含），也可以按需安装，如 `pip install -e ".[api]"`。
还没有密钥？`tenet recall` / `tenet navigate` / `tenet stats` / `tenet doubts` 配合
`EMBED_PROVIDER=local` 完全离线可用（安装 `sentence-transformers`，无任何网络调用）；
`tenet remember` / `tenet chat` / MCP 的 `learn` 工具需要真实的 `DASHSCOPE_API_KEY`
（或 `LLM_PROVIDER=openrouter`），因为它们要用一次 LLM 调用来蒸馏文本——
没有密钥时你会看到明确的 "memory write failed: ..." 报错，而不是静默失败。

更多示例见 [`examples/`](examples/) —— 零密钥演示、快速上手、助手循环、MCP 客户端、
LangChain 适配器、LangGraph `BaseStore` 适配器。

**兼容：** 任何 MCP 客户端（[Claude Desktop](examples/03_mcp_client.md)、IDE、其他智能体）·
[LangChain](examples/04_langchain_memory.py)（轻量 `TenetMemory` 适配器）·
[LangGraph](examples/05_langgraph_store.py)（`BaseStore` 适配器，见下）· 纯 HTTP
（`tenet.api:app`，`POST /chat`）。

### LangGraph `BaseStore` 适配器

Tenet 可直接作为 LangGraph 的 [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/)
——即 `StateGraph.compile(store=...)` 期望的接口——让 LangGraph 智能体的长期记忆免费获得
双时间轴取代：对同一 `(namespace, key)` 重复 `put()` 会把旧值退役进历史而不是覆盖，
与 `Tenet.ingest()` 使用的是同一套机制。

```bash
pip install tenet-memory[langgraph]
```
```python
from tenet.integrations.langgraph import TenetStore

store = TenetStore(db_path="data/agent.db")
store.put(("users", "alex"), "residence", {"city": "Montreal"})
store.put(("users", "alex"), "residence", {"city": "Toronto"})  # 取代，而非覆盖
store.get(("users", "alex"), "residence").value                # -> {"city": "Toronto"}
```
完整示例（put/get/search/delete/list_namespaces）：[`examples/05_langgraph_store.py`](examples/05_langgraph_store.py)。

### 3. 完全本地 / 离线隔离环境

写入路径中的每个调用——`ingest()` 的事实蒸馏与 `embed_texts()`——都可以指向本地模型，
因此整个闭环（学习 → 取代 → 质疑 → 时间回溯）可以在 **零云端调用**、甚至断网状态下运行：

```bash
# .env 或 shell 环境变量
LLM_PROVIDER=ollama
OLLAMA_MODEL=tenet-distiller-1.5b-v2   # 我们 LoRA 微调的蒸馏器（见下），或任意本地模型
EMBED_PROVIDER=ollama                  # 或 EMBED_PROVIDER=local 使用 bge-small（无需 ollama）
OLLAMA_BASE_URL=http://localhost:11434/v1   # 或经 Tailscale 的 GPU 机器，如 http://100.x.x.x:11434/v1
```
```bash
tenet remember "I moved from Boston to Seattle"   # 100% 本地完成蒸馏与嵌入
tenet doubts                                       # 学习型动态置信度，依然零 LLM
```

**"tenet-distiller-1.5b-v2" 是什么、测了什么：** 一个 LoRA 微调的 Qwen2.5-1.5B-Instruct，
替代云端事实蒸馏器（`qwen3.7-plus`）完成写入路径中唯一依赖 LLM 的一步——把消息转为
`subject::attribute` JSON 事实，且键的稳定性足以支撑双时间轴取代。在一份 **去污染** 的
留出评测上（全新取值与表述、与训练零重叠）：未微调的 1.5B 基座模型 **完全无法取代**
（6 个干净翻新案例中 0 个正确取代），而 LoRA 微调后的模型完全离线地复现了云端参考的
取代行为——**6/6 干净翻新全部正确取代、0.0 编造率、0.775 键一致性**
（真正驱动取代的指标——同一属性在不同表述下必须映射到同一个键）。这一键一致性
**超过了云端参考自身（0.707）**，因为训练标签对键做了强制规范化，而临场的云端提示词不会。
完整表格：[`docs/BENCHMARK.md` §10](docs/BENCHMARK.md#10-local-distiller-zero-cloud-verdict)。

**注意事项，直说：** 这些是小规模评测（n=26 条消息 / 8 个翻新组）上的确定性点估计，
没有置信区间——是探针性结果，不是生产 SLA。方向性足够强，可作为可选路径发布；
更大 N 的验证是后续工作。

训练流水线（数据生成、规范化、空目标再平衡、LoRA SFT）位于
[`scripts/distiller_lora/`](scripts/distiller_lora/)，完全可复现——
以上全部在单张 RTX 3080（16GB）上训练与评测。GGUF 文件目前在该机器上经 ollama 提供服务；
导出并自行部署：

```bash
# 在 GPU 机器上，训练（train_lora.py）+ 合并（merge_and_export.py）之后：
#   1. 将 LoRA 适配器合并进基座模型（merge_and_export.py 完成，bf16 safetensors）
#   2. 用 llama.cpp 转 GGUF —— ollama 原生 safetensors 导入会损坏合并后的
#      Qwen2.5 bf16 权重（输出乱码）；GGUF 才是真正可用的路径：
python llama.cpp/convert_hf_to_gguf.py <merged_dir> --outtype q8_0 \
    --outfile tenet-distiller-1.5b-v2.gguf
ollama create tenet-distiller-1.5b-v2 -f Modelfile   # Modelfile: FROM ./tenet-distiller-1.5b-v2.gguf
```

## 结果

LongMemEval_S —— 诚实、可复现；完整细节见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md)。

**绝对准确率取决于阅读器强度，而非记忆系统——因为检索已经饱和（recall@10 = 97.5–100%）。**
正确的事实已经在上下文里；把它们变成正确答案靠的是一个足够强的阅读器。在 **Qwen Cloud
自家阅读器**（`qwen3.7-plus`，即产品实际所用技术栈，全云端，n=100）上 Tenet 达到 **81.0%** ——
Qwen 之外的前沿阅读器结论一致（干净、非批量、逐条调用）：

| 阅读器 | n | RAG | **Tenet** |
|---|---:|---:|---:|
| **`qwen3.7-plus`**（产品 · Qwen Cloud） | 100 | 79.0 | **81.0** |
| **gpt-5.5**（前沿） | 40 | 75.0 | **77.5** |
| **Gemini-3.5-flash** | 40 | 70.0 | **75.0** |
| gpt-4o（弱阅读器效率点，见下） | 40 | 57.5 | 57.5 |

因此在有能力的阅读器下，Tenet 达到 **约 75–81%，在每个阅读器上都 ≥ 同条件 RAG**，同时
recall@10 为 **100%**、上下文比全量历史少 **98.5%** ——且在 Qwen 阅读器上*赢下*了
多会话类别（75.0 对 54.2）与时间推理类别（80.0 对 73.3）。下文你会看到的 "57.5%" 是一个
*刻意选取的弱阅读器效率工作点*，**不是** Tenet 的准确率上限；我们的头条数字曾看起来低于
Mem0/Zep 的 90%+，原因在评测所用的阅读器/嵌入器，而不是记忆设计本身。再下方的表格
在固定 gpt-4o 阅读器下刻画准确率↔token 的**帕累托前沿**。

> **注：** 发布的产品 **完全运行在 Qwen Cloud 上**（`text-embedding-v4`、`qwen3.6-flash`、
> `qwen3.7-plus`）。`gpt-4o`/`gpt-4o-mini` 在下文中**仅作为冻结的评测阅读器**出现，
> 目的是与 Mem0/Zep/MemoryAgentBench 发表时采用的协议完全对齐——与公开榜单严格同口径。

Tenet 是一条**前沿曲线，而非一个点**——一个 `expand` 旋钮即可用 token 换准确率：

| | 模式 | recall@10 | 问答准确率 | 阅读器 token | **准确率 / 1k token** |
|---|---|---:|---:|---:|---:|
| 全量上下文 | — | — | 65% | ~124,000 | 0.5 |
| RAG | top-*k* 对话轮 | 95% | 57.5% | 2,101 | 27.4 |
| **Tenet** | 效率点 | **97.5%** | 52.5% | **1,067** | **49.2** ← 每 token 最优 |
| **Tenet** | 对齐点 | **97.5%** | **57.5%** | 2,083 | 27.6 |

- **在相同或更低 token 下与强 RAG 单次准确率打平**（57.5% = 57.5%，gpt-4o）——
  信念锚定的证据扩展补上了纯信念压缩留下的差距。在 `gpt-4o-mini` 阅读器上对齐点还小幅领先
  （60.0 对 55.0）。
- 效率点上**每 token 准确率最优**（RAG 的 1.6 倍，且只用其*一半*上下文）——并且在我们测过的
  `gpt-4o-mini` 与 `gpt-4o` 阅读器间**对阅读器鲁棒**（≈1.6 倍）。
- **抗翻新（模板化基元）：** 在单属性翻新基元（§3，`bench_horizon`）上，Tenet 在每个更新
  强度下都保持 100%，而 RAG 塌至 50% ——该塌陷在 gpt-4o 阅读器下依旧成立，说明是*结构性的*，
  不是阅读器太弱。在更严酷的*同义改写版* [ChurnBench](docs/BENCHMARK.md#9-churnbench--parametric-high-churn-stress-test-measured-2026-07-10)
  （§9）上，默认技术栈（读取时一致性 + 键解析 + 时效性上下文）达到 **半衰期 32**
  （在 Qwen 技术栈上 U=32 处 98–100%）——在*保留信念历史*的同时追平了 Mem0 式"直接删除"整合。
  我们甚至把这个"直接删除"技巧移植进 Tenet（`TENET_CONSOLIDATE`），实测为**无收益**
  （与默认持平或略逊——[§9.2](docs/BENCHMARK.md#92-write-time-consolidation-tenet_consolidate--a-measured-negative-measured-2026-07-14)），因此默认关闭。
- **消融：** 仅信念–证据一致性这一条规则就把当前值准确率从 55% 提升到 100%。
- **诚实：** 仍落后于 RAG 的唯一类别是多会话综合（42.9 对 57.1，此前为 28.6）。我们如实报告。
  *（该评测在 Qwen 之外进行、单一随机种子、阅读器噪声约 ±5–7 个百分点；发布系统使用 Qwen Cloud。）*

### 🏆 标准化基准：MemoryAgentBench FactConsolidation（ICLR 2026，全部 800 题）

冲突消解——知名记忆系统失败得最惨的轴（原表：单跳 **Zep 7%、Mem0 18%、MemGPT 28%**；
多跳全部 22 个系统 **≤7%**）：

| 6K–262K 汇总 | 朴素 RAG | **Tenet** | 已发表 SOTA（mini / gpt-4o） |
|---|---:|---:|---:|
| 单跳 | 47.8 | **86.5** [82.8, 89.5] | 78.0 / 94.8 |
| 多跳 | 4.5 | **30.2** [26.0, 34.9] | 30.2 / 51.5 |

**超过已发表的 mini 档单跳 SOTA、多跳与之持平——用的是本地 7B 骨干模型和*零 LLM* 的
确定性写入。** SubEM 与官方提示词逐字一致；Wilson 置信区间；无长度塌陷（每个干草堆规模下
单跳 ≥81%）。详见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md) §6。

**MAB Accurate-Retrieval**（约 2,000 题，上下文 197K–534K token，官方逐基准指标，
同口径 gpt-4o-mini 阅读器）：AR 平均 **59.3** ——仅次于 HippoRAG-v2（65.1，它对每个上下文
token 跑 LLM 开放信息抽取；Tenet 的写入**只用嵌入向量**），比 Mem0（32.6）/ Zep（37.5）/
MemGPT 高 20 分以上，并且**在 EventQA 上胜过全场（70.7 对 67.6，置信区间不相交）**。
RULER 多跳是诚实的失分项（45 对 66）。详见 [`docs/BENCHMARK.md`](docs/BENCHMARK.md) §7。

## 智能体

Tenet 附带一个运行在 Qwen Cloud 上的个人助手（[`src/tenet/agent.py`](src/tenet/agent.py)）：
```
you › Hi! I'm Alex, I live in Montreal and work as a data analyst.
assistant › Nice to meet you, Alex! How's the analyst work in Montreal?   [remembered 2 facts]
… 数周后 …
you › I moved to Toronto and got promoted to senior analyst!
you › Where do I live and what's my job now?
assistant › You live in Toronto and you're a senior analyst. Congrats on the promotion!
```
```bash
python -m tenet.agent          # 交互式助手（或：tenet-agent）
python scripts/demo_agent.py   # 脚本化剧情（视频演示用）
```

## 架构
![architecture](docs/architecture.svg)

一个双时间轴存储（信念 + 证据）之上的两层结构、两个接口（MCP + HTTP），
由 Qwen Cloud（阿里云百炼 / Model Studio）驱动。单页组件图 + 核心公式 +
"仅注解不变量"的论证见：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。
最初的设计范围：[`docs/DESIGN.md`](docs/DESIGN.md)；与 Mem0/Zep/Letta/Mastra 的定位对比：
[`docs/COMPARISON.md`](docs/COMPARISON.md)。

## 复现论文

每个基准都是一条 CLI 命令——provider 预设 + 配置 + git-sha 记录到
`data/bench_runs.jsonl`。`tenet bench run` 直接调度产生论文数字的
`scripts/bench_*.py`（唯一事实来源），绝不重新实现。
```bash
tenet bench list                        # 全部基准 + 各自复现哪张图/哪一节
tenet bench run <name> --dry-run ...     # 打印确切命令+环境，不运行
tenet bench results                     # 历史运行结果表

python scripts/test_memory.py ; python scripts/test_tenet_e2e.py                  # 能力测试
tenet bench run churn --provider ollama --principals 12 --k 6 --updates 2,4,6,8,10,12   # 图 1 翻新
tenet bench run lme-recall --provider openrouter --limit 40 --k 10 --seed 2 --qa            # 效率点
tenet bench run lme-recall --provider openrouter --limit 40 --k 10 --seed 2 --qa --expand 20  # 对齐点
tenet bench run knowledge-update --provider ollama --principals 4                 # 取代消融
```
`--provider` 预设：`ollama`（完全离线：本地嵌入 + qwen2.5:7b 阅读器）、
`openrouter`（本地嵌入 + gpt-4o-mini 阅读器）、`local`（仅嵌入）、
`qwen`（Qwen Cloud）。完整矩阵 + 读取路径性能分析：[`docs/BENCHMARK.md`](docs/BENCHMARK.md)、[`docs/HARNESS.md`](docs/HARNESS.md)。

## 仓库结构
```
paper/tenet.md tenet_full.pdf   论文（2 页版 + 完整预印本）
src/tenet/  core.py memory.py distill.py config.py   信念状态记忆引擎
            navigate.py                               自适应无 LLM 多跳召回
            agent.py                                  助手
            mcp_server.py api.py alicloud_oss.py      接口 + 阿里云部署
            integrations/langgraph.py                 LangGraph BaseStore 适配器
examples/   00_zero_key_demo.py 01_quickstart.py 02_assistant.py 04_langchain_memory.py
            05_langgraph_store.py                     零密钥演示、快速上手、助手循环、
                                                        LangChain + LangGraph 适配器
scripts/    demo_agent.py    视频演示
            bench_horizon.py bench_factcon.py bench_mab_ar.py lme_recall.py   基准
            test_memory.py test_dynamics.py test_agent_uncertainty.py test_errors.py
            test_langgraph_store.py test_navigate.py test_tenet_e2e.py smoke_test.py   测试
docs/ BENCHMARK.md COMPARISON.md DESIGN.md DEPLOY.md  architecture.svg horizon.svg
```

## 引用
```bibtex
@misc{tenet2026,
  title  = {Tenet: Agent Memory as a Self-Consistent Belief State},
  author = {Anas},
  year   = {2026},
  note   = {Global AI Hackathon with Qwen Cloud, Track 1},
  url    = {https://github.com/Nas01010101/tenet}
}
```

## 缘起
Tenet 始于 [Global AI Hackathon with Qwen Cloud](https://qwencloud-hackathon.devpost.com)
（Track 1: MemoryAgent）参赛项目——黑客松材料位于 [`docs/hackathon/`](docs/hackathon/)。

## 许可证
MIT —— 见 [LICENSE](LICENSE)。

---

<sub>翻译说明：本文件为 [README.md](README.md) 的简体中文翻译，由 Anthropic 的大语言模型
Claude（Fable 5）翻译，并用开源神经机器翻译系统
[Argos Translate](https://github.com/argosopentech/argos-translate)（OpenNMT/CTranslate2 栈，
LibreTranslate 的翻译引擎）做 zh→en 回译交叉校验；全部数字、模型名与命令经脚本逐一
与英文原版比对（见 `scripts/check_translation.py`）。若与英文版有出入，以英文版为准。</sub>
