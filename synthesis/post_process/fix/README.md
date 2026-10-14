# i2i trajectory 增强：Selector + Writer 循环

当前入口：`python -m synthesis.post_process.fix.run`。
算法版本 `i2i-fix-v4-selector-writer`，旧运行目录不可续跑，必须使用新 workdir。
输入不修改；改写通过结构校验和最终语义审核后才替换输出中的对应记录。

## 当前流程

1. `progress.py`：判断原 i2i_search 是否产生关键进展。无进展保留原样。
   提取所有实际引用/依赖的条目，不再限制 1–2 个 reference；记录结论前的真实看图。
   只有实际返回图片的工具结果才算看图，纯网页文字不算。
   `anchor_image_ids` 是采用的搜索条目（只用标题/网页也必须填写），与实际看图
   的 `viewed_before_conclusion_ids` 分开。非法字段默认局部修复一次
   （`progress_audit_retries=1`），不重跑搜索；真正无进展/不安全的判断不作格式重试。
   同一图片多次读取合法：记录所有读取位置，首次读取作为复用衔接点，后续观察保留。
2. `replay_i2i.py` / `visual_evidence.py`：重搜 top20，下载并并行分类每张候选。
   已有图片直接复用；缺失的历史引用通过 `title_recovery.py` 真实标题搜索恢复。
   同来源恢复允许 SAME_ENTITY；未看图的辅助引用允许保留原网页的 RELATED、SIMILAR、UNRELATED 配图，分类不变，不能冒充正例。跨来源借图仍要求 CONSISTENT，避免伪造图片和网页的对应关系。
   最终候选列表按 seed 和候选集合确定性打散，不沿用组卡模型的正例优先排序，也不强制错误选择。上下文编辑读取完整原轨迹及明确插入位置，只改写所有待插入片段；旧排名、被移除候选评论、过时的一致性判断和原有过程措辞通过 `deferred_context_issues` 交给全局编辑，不再提前拒绝。需要虚构证据或改变动作/答案才能成立的实质矛盾仍拒绝。
   组卡返回非法 ID、重复条目、缺少引用或正例时，`composition_retries` 默认允许一次局部纠错，仅重调组卡 LLM，不重跑搜索和验图，不模糊匹配 ID。成功纠错记录在计划的 `composition_repairs`；持续无效则隔离并报告具体校验原因。
3. `search_cards.py`：模型构建最多五条真实结果，全部保留历史依赖；其余从新的
   搜索结果补充。至少一条 CONSISTENT 或 SAME_ENTITY，优先 CONSISTENT。
   不足五条可继续；无法恢复必需引用、资源冲突或没有正例则明确跳过。
   增强组卡通常最多两条正例（CONSISTENT + SAME_ENTITY）；必需历史正例超过两条时全部保留，不额外加入可选正例。剩余位置尽量使用 RELATED / SIMILAR，不足时用 UNRELATED 补齐；若仍不足五条，不再增加正例凑数。UNRELATED 不可被 Selector 选择。正例限额和优先负例数量由程序校验。
4. 限制正例前，若至少五条可用候选全部属于 CONSISTENT / SAME_ENTITY，视为自然收敛并保留原轨迹；否则按上述增强组卡规则处理。必需引用已占满五个正确槽位时，也保留收敛行为。
   不足五条不自动视为“五条收敛”；不是标题相似就跳过。
5. `rollout.py`：先为每个可选卡片生成基于可见文本的打开理由，不进行竞争力淘汰。
   Selector 选择一条 → 使用真实缓存图片 → Writer 对照原图写判断 → 将这段
   选择理由和检查文字加入历史 → Selector 从剩余条目继续选择。
   第一轮只提供题目、截至搜索调用的历史、卡片和候选理由；后续增加实际检查历史。
   不给 Selector 分类、答案、原采用条目、未来推理或未打开的候选图。
6. 选对后结束。原轨迹未正确看图：补完正确图检查再衔接。
   原轨迹已正确看图且选中同条目：保留原看图工具调用和 observation，不重复生成。
   已看图但选中另一条正确结果：`different_correct_candidate`，本次增强跳过。
7. `context_patch.py` 统一改写所有新增文字，原轨迹仅作只读上下文，输出完整 `trace_patches`，禁止输出原消息补丁。每段保持 agent 解题视角，禁止透露编辑、插入或增强过程。`splice.py` 将上一轮 after 与下一轮 before 合并，最后的 after 前置到原接入段，原文保留；最终审核仍检查完整拼接结果。
   原工具结果、工具调用和答案保留；不能把原看图之后的证据移动到看图之前。
8. 拼接和结构校验成功后，`global_refine.py` 接收完整轨迹与全部可用图片，仅给 assistant turn 添加连续编号。每轮都必须返回检查结果；可以改写任意 assistant 推理，统一全局逻辑、衔接和文风，但保留工具调用、观察、消息顺序和最终答案。返回严重跨轮逻辑错误时记录 `global_refinement.severe_errors`、`failure_kind=global_logic_error` 和 `discard=true`，从新导出数据中自动排除，不受 `--drop-quarantined` 开关影响；检查点仍保留原始记录供排查。
9. 全局改写后再次进行结构校验，再进入原 `validate.py` 的带图片最终语义审核。新增阶段用 `model`，最终审核仍用 `judge_model`；`global_refine` 单独记录耗时。一次成功处理通常增加一次 LLM 调用，输出格式错误才会局部重试。

全局编辑输入明确标记 `updated_search_results`，列出后续 `affected_assistant_ids` 并接收转交问题（原消息位置转换为全局 assistant 编号）。其 prompt 包含 Getty、列车搜索排名、营地候选评论和 abbreviated history 四类示例；仅修正推理能维持真实执行路径时修改，实体偷换等严重错误不能借修辞掩盖。输出后再次扫描原有和改写后的推理，不能以 reasoning=null 留下已知过程泄漏。

direct / contrast 只是事后统计标签，不是提前分配的方案。
Selector 额外报告 equally_plausible_ids、preference_evidence 和 exclusions；文本排除须引用题目/已观察历史中的证据，不能用未知国家排除其他国家。多个同样值得检查的候选按 seed、题目、当前步数和候选 ID 可复现地抽选，不读取分类标签，使用所选候选此前生成的检查理由，不新增 LLM 调用。selection_assessments 记录原始判断，selection_modes 中 plausible_tie_sample 标识探索选择。插入编辑与全局编辑禁止把优先级差异强化成未经证实的错误排除。
不再使用竞争力模型、错误次数配额、第三轮指定正确图或自动 recovery 重搜分支。
最多检查当前列表中的可选条目各一次（最多五次），找到正确项即停止。
无法可靠判断时不编造结论，保留原样或隔离失败增强。

## 分类与证据

| 内部类别 | 含义及用途 |
| --- | --- |
| CONSISTENT | 同图内容，允许裁切、缩放、旋转；可作为正确检查终点 |
| SAME_ENTITY | 确认同一主体但不是同一照片；可作为身份检查终点 |
| SIMILAR | 文本不能排除，视觉相似但确认不同主体；可选择并排除 |
| RELATED | 文本不能排除，视觉相似度低且确认不同主体；可选择并排除 |
| UNRELATED | 仅卡片文本已有充分理由排除；可作背景，Selector 不可选 |

代码沿用 `CONSISTENT` 拼写，对应讨论中的 CONSISTENCY。
`DIFFERENT_PHOTO`（身份不确定）及 `UNRESOLVED` 是安全辅助状态，不冒充正例或负例，
不进入新可选列表。必需引用落入这些状态则跳过。
Writer 必须给出目标相关、可见的成对比较；衣服、姿态等差异不能单独证明不是同一人。
SAME_ENTITY 不能写成同一照片，也不能把活动/日期/地点从另一照片直接迁移。
看图匹配不等于验证来源网页内容。不可变答案与 SAME_ENTITY 证据矛盾时拒绝增强。
标题恢复保留原来源页面引用，同时在私有审计保存实际检索来源；不伪造下载资源。

## 模块与 prompt

| 文件 | 职责 / prompt |
| --- | --- |
| `progress.py` | 原搜索进展、引用和看图时间；`PROMPT` |
| `visual_evidence.py` | 并行实际图像分类；`PROMPT` / `CLASSIFICATION` |
| `search_cards.py` | 保留引用、组装最多五条；`COMPOSE` |
| `rollout.py` | 候选打开理由 `REASONS`，自主选择 `SELECT`，顺序执行循环 |
| `rewrite.py` | 当前图像比较文字；`AFTER` / `write_after` |
| `answer_consistency.py` | SAME_ENTITY 与不可变答案冲突检查；`PROMPT` |
| `context_patch.py` | 连贯性与文风微调；`PROMPT` |
| `validate.py` | 结构检查、最终多模态语义审核；`JUDGE` |
| `pipeline.py` | 编排上述模块，记录每阶段审计和耗时 |

旧 `candidate_sets.py`、`dual_reference.py`、`eligibility.py`、`recovery.py`、
`selection.py` 的旧规划/选择函数保留供历史回归测试；主流程不调用。
`candidate_relevance.py` 的竞争力 prompt 不再调用，只复用公开卡片格式辅助函数。
`rewrite.py` 的旧批量 `write_trace` 不再调用：Writer 必须先完成本轮才能选择下一张。

## 配置与并发

以 `config.example.json` 为准。当前三个模型别名均为 `gemini38flash_internal_azure`：
`model` 用于进展、组卡、Selector、Writer 和上下文；`verifier_model` 分类图片；
`judge_model` 最终审核。相同别名的独立调用不等于不同模型交叉验证。
默认关闭额外盲审，保留最终语义审核；显式开启 `blind_visual_review` 则在最终阶段额外盲比对。

- `record_workers=16`：跨记录并行，可用 CLI 覆盖。
- `visual_check_workers=16`：每条记录下载/分类并发。
- `global_visual_workers=128`：全进程下载/分类总并发上限。
- 单条记录 Selector 与 Writer 严格串行；不同记录之间仍并行。
- `result_count=5` 是最多条数，不强制凑满。
- `max_windows_per_record=1`：默认每条记录处理一个窗口。

旧配置字段为兼容读取仍存在，但新流程不使用 branch_weights、wrong_count_weights、
replacement_counts、positive_slots、convergence_min、similar_negative_probability、
max_original_views、max_selection_checks、retry_strategies、writer_workers、embedding 配置。
这些字段不能再控制错误检查次数或跳过概率。

## 小批测试

```bash
python -m dotenv -f synthesis/.env run -- \
  python -m synthesis.post_process.fix.run \
  --input data/sharegpt_dataset_refined_gpt54_only_17k_final/trajectories_sharegpt.json \
  --workdir synthesis/.ignore/i2i_fix_selector_writer_v1_10 \
  --config synthesis/post_process/fix/config.example.json \
  --mode augment --limit 10 --selected-only \
  --output synthesis/.ignore/i2i_fix_selector_writer_v1_10/trajectories_sharegpt.json
```

`--limit` 是输入记录数，不是成功改写数。`--mode index` 只解析，不调用网络，
其中窗口标记 requires_progress_audit，不能离线断言语义上可增强。
配置/代码变更须新建 workdir；同一 workdir 不要运行多个进程。
同配置同代码可续跑，`--retry-quarantined` 重试隔离记录。
可用 `--llm-cache-dir` 跨 workdir 复用完全相同的模型请求；不要多进程同时写共享缓存。

## 输出、计时和验证

`records/<index>/window_*_audit.json` 保存进展判定、每张图分类、最终列表、
逐轮 reasoning、步骤、跳过原因及最终审核；部分循环失败时也保留已完成步骤。
`records/<index>.json` 是记录级提交点；`audit.json` / `summary.json` 汇总最终状态、
实际错误排除次数和 `skip_reasons`。跳过不表示数据坏；quarantined 表示本次增强失败。
默认输出仍包含跳过和隔离记录的原文；`--drop-quarantined` 才排除隔离记录。

图片保存在仓库相邻的 `../tmp/i2i_fix_images`，不是系统 `/tmp`。
输出使用真实图片路径，训练期间必须保留新增图片和原数据图片。

`timing.jsonl` / `timing_runs` / summary 中记录墙钟时间、阶段、缓存和失败耗时。
计时事件由工作线程入内存队列，同一路径共享一个后台写入器，持久打开文件，每 128 条或约 1 秒批量写入并刷新；磁盘 I/O 不阻塞业务线程。正常结束及异常退出通过 finally 排空并关闭，写入失败或排空超过 30 秒明确报错。运行中日志可能短暂滞后；强制 kill/进程崩溃仍可能丢失尚未落盘的队列事件。直接使用 Timer 的调用方需在所有生产线程结束后调用 close()。高峰积压使用内存队列，不通过磁盘背压拖慢工作线程。
新阶段包括 progress、composition、inspection_reasons、selector、writer_observation、
selection_writer_loop、context、review。父阶段包含子阶段，不能把所有秒数相加。

```bash
python -m unittest discover -s synthesis/post_process/fix/tests -q
ruff check synthesis/post_process/fix
```

离线测试验证状态流转、信息边界、工具/图片保留和失败回退，不能代替真实模型小批质量检查。
# Export policy and resume

Dependency wiring (v35): the active `assess_progress` request now requires a
`dependency_image_ids` list for additional downstream title/snippet/URL-keyword
evidence. Its validated IDs are merged with adopted anchors, original image
reads, tool resource dependencies, and explicit assistant prose resource
citations. Both recovery and composition receive the full preservation set as
`anchor_image_ids` / `required_reference_ids`. This adds no separate LLM request;
malformed dependency IDs use the existing bounded progress-audit repair.

Execution repairs (v34): the initial splice still preserves its prefix exactly.
After global refinement, assistant reasoning may change across the full record;
validation compares against a pre-refinement snapshot and preserves all message
metadata, non-assistant messages, tool-call/answer blocks and image paths/counts.
Candidate inspection-rationale schema/ID failures retry locally using
`context_patch_retries` (default one retry), with exact allowed IDs and validation
feedback. Attempts are saved in `plan.metadata.inspection_reason_attempts`.
No search/download/classification is repeated by this local retry. Persistent
format failures remain retry-pending rather than being silently repaired.

Visual-evidence guardrails (v33): retained historical cards preserve their public
snippets and URL keywords. Anchor analysis also identifies downstream evidence
dependencies; explicit image/page references are protected programmatically.
These dependencies can occupy more than two slots, overriding negative quotas.
The one/two identity-anchor policy is unchanged.

Writer returns a structured local-goal assessment alongside natural reasoning.
The sequential loop accepts a positive only when the Writer supports the local
goal, and accepts a negative exclusion only when the Writer supports rejection.
Uncertainty or disagreement with the planned category produces a quality rejection
(no forced justification, no automatic relabeling, no fallback export). Malformed
assessment fields get bounded local retries; substantive evidence conflicts do not.
This is conservative and can reduce yield, including when the original was sound.
Inspection findings and unresolved questions are passed to both editing stages;
the full-image global editor and final reviewer remain required. These guardrails
do not guarantee that model-generated visual judgments are correct.

The exporter now writes only completed, accepted records into the new SFT dataset:

- `augmented`: export the accepted augmented trajectory.
- `keep_original`: normal skips (including no eligible window or convergent results) retain the original. This is not a whole-record quality certification.
- `quality_rejected`: semantic review rejection or severe global logic failure; exclude from export and record the audit/reasons in `quality_rejected.json`.
- `retry_pending`: execution, tool, selection-schema or structural-validation failure; exclude from export and record the audit/reasons in `retry_pending.json`.

These dispositions are record-level: a failed window prevents a partially successful multi-window record from being exported. Unprocessed records are also excluded, rather than silently copied into the output. The summary lists their indices. A quality rejection excludes the whole record conservatively, even when the defect might be limited to the rewrite.

Use the same command and workdir with `--resume` to reuse completed checkpoints, continue missing records, and retry execution failures once per invocation. Quality rejections are terminal for that manifest and are not retried. `--retry-quarantined` now has the same execution-failure-only retry policy. Successful tool/search caches are retained; a retry uses a new per-record LLM cache namespace so a cached invalid response cannot trap the retry. Record checkpoints are written atomically, and the export is rebuilt atomically at the end of a completed run; interrupted runs keep their completed checkpoints.

By default, input, config and code must match the workdir manifest. For an
intentional code-only migration, use `--resume --resume-allow-code-change`.
Input path/content, semantic configuration, and schema version must still match.
With `--resume`, concurrency-only changes to `record_workers`, `visual_check_workers`,
`writer_workers`, and `global_visual_workers` are allowed and recorded in
`manifest.json` under `config_history`. For example, increase `--record-workers 16`
to `--record-workers 32`; no configuration-change override is needed. Other fields
(including model, seed, and candidate policy) remain strictly checked. Stop the
previous process before restarting against the same workdir. Installing this
resume-policy code change still requires `--resume-allow-code-change` once.
This opt-in reuses completed old-code checkpoints WITHOUT
revalidating them; quality rejections remain terminal, execution failures retry,
and missing records use the current code. `manifest.json` records `code_history`
and the legacy checkpoint version atomically before processing. Checkpoints and
the exported audit carry per-record `code_digest`; `summary.json` reports
`record_code_digests`. New/retried LLM requests use code-isolated caches, while
search/download caches remain reusable. Legacy checkpoints gain provenance only,
not new trajectory content. Subsequent same-code resumes need only `--resume`.
Do not run different code versions concurrently against the same workdir.

To process the full dataset after a pilot, remove `--limit`/`--indices` and use
`--resume`; leaving those selectors in place only resumes that subset.
`--selected-only` continues to limit export to the requested subset. Existing
`--drop-quarantined` and `--drop-failure-kinds` remain accepted, but failures are
now excluded by default.
