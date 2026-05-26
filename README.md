# LHTIR: verl 0.5.0 多轮工具训练扩展

## 项目总览

本仓库基于当前目录下的 `verl/`（verl 0.5.0 版本）扩展了一套面向 FTRL / MatchTIR 风格任务的多轮工具调用训练链路。核心改动集中在 `prog_env/`，并通过少量 `verl/` 接入点把自定义 dataset、agent loop、reward manager、advantage estimator 和 policy loss 挂到原有 PPO trainer 中。

主要目标：

- 支持每条训练样本携带自己的工具 schema 与工具实现代码；
- 在 rollout 阶段通过 agent loop 解析模型工具调用，并用样本内 `codes` 模拟工具执行；
- 支持多轮工具调用轨迹的过程奖励、turn-level advantage 与自定义 loss；
- 保持 `python -m verl.trainer.main_ppo` 的启动方式，尽量通过 Hydra override 控制实验配置。

## 目录与模块说明

- **`prog_env/agent_loop/`**：自定义异步 agent loop。
  - `agent_loop.py`：从 verl agent loop 改造而来，支持 `tools/codes` 从 batch 传入，输出 `__step_length__` 等额外信息。
  - `tool_agent_loop_3.py`：当前主要使用的工具 agent，注册名为 `tool_agent_3`，支持代码模拟工具、`max_step_length`、overlong mask、progressive turns。
  - `tool_parser.py`：工具调用解析器，新增 `hermes_prog`，用于处理 `<think>...</think>` 后的 `<tool_call>...</tool_call>`。

- **`prog_env/dataset/`**：自定义 RLHF dataset。
  - `rl_dataset.py`：读取 parquet 中的 `messages`、`tools`、`codes` 等字段，构造带 tool schema 的 chat template，并把 `tools` 传给 rollout。

- **`prog_env/reward_manager/` 与 `prog_env/reward_score/`**：自定义奖励。
  - `reward_manager/matchtir.py`：注册 `matchtir` reward manager，负责从 `DataProto` 中读取响应、样本字段并调用 reward function。
  - `reward_score/matchtir.py`：包含工具调用匹配、过程奖励、answer F1/EM/SubEM 等打分函数。

- **`prog_env/advantage_est/`**：自定义 advantage estimator。
  - `mathtir_adv.py`：注册 `mathtir`、`mathtir_fast`、`mathtir_fast_reverse`，用于多轮/turn-level reward 到 token-level advantage 的转换。

- **`prog_env/loss_cal/`**：自定义 policy loss。
  - `turn_loss.py`：注册 `gtpo`，按 turn 聚合 importance ratio。
  - `turn_loss_2.py`：注册 `gtpo_test`。
  - `aspo_loss.py`：注册 `aspo`。
  - `gspo_loss.py`：包含 `gspo` 实现；注意当前 `prog_env/loss_cal/__init__.py` 未默认导入它，如需使用应同步注册导入。

- **`prog_env/config/`**：项目 Hydra 配置。
  - `ftrl_multiturn.yaml`：继承 verl PPO trainer 配置，打开 sglang multi-turn，并设置 `return_raw_chat=True` 等基础项。

- **根目录训练脚本**
  - `run_qwen3-4b_ftrl_multiturn*.sh`：不同实验配置的启动脚本。
  - 输出默认写入 `checkpoints/`、`rollout/`、`logs/`、`wandb/`。

## 相对 verl 0.5.0 的核心调整

### 1. 自定义 Dataset 接入

verl 侧 `verl/trainer/main_ppo.py` 支持通过以下 override 加载外部 dataset：

```bash
data.custom_cls.path=$PROJECT_DIR/prog_env/dataset/rl_dataset.py
data.custom_cls.name=RLHFDataset
```

`prog_env/dataset/rl_dataset.py` 相比默认 `RLHFDataset` 的关键变化：

- 新增 `tools_key`，默认从 parquet 的 `tools` 字段读取工具 schema；
- 在 prompt 长度过滤和实际 tokenization 时调用 `tokenizer.apply_chat_template(messages, tools=tools, ...)`；
- 将 `tools` 放入 `row_dict["tools"]`，供 agent loop 在 rollout 时使用；
- 支持 `enable_thinking`，用于控制 Qwen 类模型 chat template 中的 thinking 行为。

### 2. 自定义 Async Agent Loop

verl 侧 `verl/trainer/ppo/ray_trainer.py` 支持通过 `actor_rollout_ref.rollout.custom_cls` 加载外部 rollout manager：

```bash
+actor_rollout_ref.rollout.custom_cls.path=pkg://prog_env.agent_loop
+actor_rollout_ref.rollout.custom_cls.name=AgentLoopManager
```

`prog_env/agent_loop/` 的主要调整：

- `AgentLoopWorker.generate_sequences` 从 batch 中读取 `tools` 和 `codes`；
- 通过 `AGENT_NAME` 环境变量或样本内 `agent_name` 选择 agent loop，当前常用 `tool_agent_3`；
- `ToolAgentLoop_3` 不再依赖统一 `tool_config_path`，而是使用样本内 `tools` 生成 prompt；
- 模型产生 `<tool_call>...</tool_call>` 后，使用样本内 `codes` 执行对应 Python 函数并把结果作为 tool response 追加到上下文；
- 新增 `max_step_length`，限制单个 assistant turn 的生成长度；
- 新增 `enable_overlong_mask` 与 `overlong_mask_scope`，用于过长轨迹的 loss mask 控制；
- 新增 progressive turns，可随 global step 逐步增加最大 assistant/user turns；
- 记录每个 turn 的生成长度，并通过 `__step_length__` 和 metrics 输出。

### 3. 工具解析格式

`prog_env/agent_loop/tool_parser.py` 新增 `hermes_prog` parser：

- 只解析 `</think>` 之后的 `<tool_call>...</tool_call>`；
- 要求当前 assistant step 以 tokenizer 的 EOS token 结束，避免截断时误解析工具调用；
- 解析出的工具调用统一转换为 `FunctionCall(name, arguments)`。

常用配置：

```bash
actor_rollout_ref.rollout.multi_turn.format=hermes_prog
```

### 4. Reward Manager 与 Reward Function

verl 侧 `verl/workers/reward_manager/__init__.py` 导入 `prog_env.reward_manager`，使 `matchtir` 可以被 registry 找到：

```bash
reward_model.reward_manager=matchtir
custom_reward_function.path=$PROJECT_DIR/prog_env/reward_score/matchtir.py
custom_reward_function.name=compute_process_KM
```

`MatchTIRRewardManager` 会从每条样本中读取：

- 模型响应；
- `codes`；
- `unsolved_set`；
- `solve_rate`；
- `answer`；
- 训练阶段的 `ground_truth` 工具调用。

常见 reward function：

- `compute_process_KM`：过程级奖励，结合工具调用匹配和 answer reward；
- `compute_answer_f1_recall`：从 `<answer>...</answer>` 抽取答案并计算 F1；
- `compute_answer_em` / `compute_answer_subem`：answer-level EM / SubEM；
- `compute_solve_f1`、`compute_toolrl` 等：工具调用与解题效果相关奖励。

### 5. 自定义 Advantage Estimator

verl 侧 `verl/trainer/ppo/ray_trainer.py` 支持非内置 `algorithm.adv_estimator`，并通过 import `prog_env.advantage_est` 完成注册。

常用配置：

```bash
algorithm.adv_estimator=mathtir_fast
+algorithm.step_adv_mode=forward
```

已注册 estimator：

- `mathtir`：基于多轮 response mask 切分 turn，融合 group-level 与 step-level advantage；
- `mathtir_fast`：更快版本，常用于当前训练脚本；
- `mathtir_fast_reverse`：反向/变体实现，用于实验对比。

### 6. 自定义 Policy Loss

verl 侧 `verl/trainer/ppo/core_algos.py` 在找不到内置 loss 时会 import `prog_env.loss_cal`，利用注册器加载自定义 policy loss。

常用配置：

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=gtpo
actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean"
```

当前默认导入并注册：

- `gtpo`：turn-level importance ratio；
- `gtpo_test`：GTPO 实验版本；
- `aspo`：ASPO 实验版本。

如果要使用 `gspo`，需要确保 `prog_env/loss_cal/gspo_loss.py` 被导入注册。

## 数据格式要求

训练脚本默认使用：

```bash
data.train_files=$PROJECT_DIR/data/ftrl/with_agent_name/train.parquet
data.val_files=$PROJECT_DIR/data/ftrl/with_agent_name/test.parquet
```

parquet 至少应包含以下字段：

- `messages`：chat messages，脚本中配置 `data.prompt_key=messages`；
- `tools`：JSON 字符串，表示当前样本可用工具 schema；
- `codes`：JSON 字符串或可反序列化对象，表示工具名到 Python 函数代码的映射；
- `answer`：最终答案，用于 answer-level reward；
- `ground_truth`：训练 split 下的目标工具调用序列；
- `unsolved_set`：工具名到待解决 answer 集合的映射；
- `solve_rate`：样本已有解题率或先验成功率；
- `split`：`train` 或 `test`；
- `data_source`：reward manager 默认使用的 reward key。

可选字段：

- `agent_name`：指定样本使用的 agent loop；若设置了环境变量 `AGENT_NAME`，会优先用环境变量覆盖；
- `extra_info.index`：用于 GRPO / 多采样分组；
- `extra_info.tools_kwargs`、`extra_info.interaction_kwargs`：保留给 rollout 侧扩展。

## 运行方式

### 环境准备

训练脚本中包含机器相关路径，运行前需要按实际环境确认：

```bash
PROJECT_DIR="$(pwd)"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/verl:$PYTHONPATH"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRITON_CACHE_DIR="/ssd/tmp/triton_cache_$(whoami)"
export RAY_DEBUG_MODE="0"
export NCCL_IB_TIMEOUT=22
export NCCL_TIMEOUT=9999999999
```

还需要确认：

- conda 环境名称是否正确，例如 `verl_5_2` 或 `vel_5`；
- `MODEL` 路径是否存在，例如 `/root/model/Qwen3-4B-Thinking-2507`；
- parquet 数据路径是否存在；
- 单机 GPU 数与 `trainer.n_gpus_per_node`、`CUDA_VISIBLE_DEVICES` 一致。

### 常用启动命令

Debug / 小 batch 脚本：

```bash
bash run_qwen3-4b_ftrl_multiturn_debug.sh
```

常规多轮训练脚本：

```bash
bash run_qwen3-4b_ftrl_multiturn.sh
```

并行/8 GPU 实验脚本：

```bash
bash run_qwen3-4b_ftrl_multiturn_bingxing.sh
bash run_qwen3-4b_ftrl_multiturn_bingxing2.sh
bash run_qwen3-4b_ftrl_multiturn_bingxing3.sh
```

### 脚本差异

| 脚本 | 主要用途 | Advantage | Policy loss | Reward function | 关键 rollout 配置 |
| --- | --- | --- | --- | --- | --- |
| `run_qwen3-4b_ftrl_multiturn_debug.sh` | debug / 小 batch | `mathtir_fast` | 默认 | `compute_process_KM` | `AGENT_NAME=tool_agent_3`，`n=16`，`max_step_length=4096` |
| `run_qwen3-4b_ftrl_multiturn.sh` | 常规训练 | `grpo` | 默认 | `compute_answer_f1_recall` | `n=8`，`max_step_length=2048`，`AgentLoopManager` |
| `run_qwen3-4b_ftrl_multiturn_bingxing.sh` | 并行基线实验 | `mathtir_fast` | `vanilla` | `compute_process_KM_prog_hermes` | `hermes_prog`，`n=8`，overlong mask 关闭 |
| `run_qwen3-4b_ftrl_multiturn_bingxing2.sh` | GTPO + overlong mask 实验 | `mathtir_fast` | `gtpo` | `compute_process_KM` | `enable_overlong_mask=True`，`overlong_mask_scope=trajectory` |
| `run_qwen3-4b_ftrl_multiturn_bingxing3.sh` | progressive turns 实验 | `mathtir_fast` | `vanilla` | `compute_process_KM` | `progressive_turns_enabled=True`，4 到 12 turns |

`bingxing2` 的关键 loss 配置：

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=gtpo
actor_rollout_ref.actor.clip_ratio_low=2e-3
actor_rollout_ref.actor.clip_ratio_high=2e-3
actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean"
```

`bingxing3` 的 progressive turns 配置：

```bash
+actor_rollout_ref.rollout.multi_turn.progressive_turns_enabled=True
+actor_rollout_ref.rollout.multi_turn.progressive_start_turns=4
+actor_rollout_ref.rollout.multi_turn.progressive_end_turns=12
+actor_rollout_ref.rollout.multi_turn.progressive_turn_scale_step=8
+actor_rollout_ref.rollout.multi_turn.progressive_turn_scale_delta=4
+actor_rollout_ref.rollout.multi_turn.progressive_sync_user_turns=True
```

## 测试与检查

本项目训练效果检查主要参考 `scripts/train/merge_binxing.sh` 和 `scripts/evaluate/*_binxing.sh`。典型流程是先把 verl FSDP actor checkpoint merge 成可推理模型，再用 vLLM 起 OpenAI-compatible server，最后运行 ToolHop / FTRL / BFCL 评测脚本。

### 1. Merge 训练 checkpoint

训练产物默认在 `checkpoints/<project>/<experiment>/global_step_xx/actor`。评测前先参考 `scripts/train/merge_binxing.sh` 合并 actor：

```bash
bash scripts/train/merge_binxing.sh
```

脚本中的关键变量需要按实验修改：

- `ACTOR_DIR`：指向训练保存的 `global_step_xx/actor`；
- `TARGET_DIR`：指向合并后的 HuggingFace 格式模型目录，通常放在 `checkpoints/merged_checkpoints/`。

当前 `merge_binxing.sh` 示例：

```bash
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $ACTOR_DIR \
    --target_dir $TARGET_DIR
```

### 2. ToolHop / FTRL 风格评估

参考 `scripts/evaluate/eval_toolhop_vllm_binxing.sh`：

```bash
bash scripts/evaluate/eval_toolhop_vllm_binxing.sh
```

该脚本会：

- 使用 `vllm serve` 加载 `checkpoints/merged_checkpoints/...` 下的模型；
- 监听 `PORT=7902`，served model name 为 `MatchTIR`；
- 对 `Free`、`Direct`、`Mandatory` 三个 ToolHop scenario 逐个运行；
- 调用 `evaluate/evaluation_toolhop.py`；
- 输出 JSONL 到 `results/ToolHop/Qwen3-4B/`；
- 输出日志到 `results/ToolHop/Qwen3-4B/` 和 `logs/vllm_logs/`。

核心评估参数：

```bash
python3 $PROJECT_DIR/evaluate/evaluation_toolhop.py \
    --scenario ${scenario} \
    --series qwen \
    --model_path ${model} \
    --input_file $PROJECT_DIR/data/toolhop/ToolHop.jsonl \
    --output_file ${save_file} \
    --enable_thinking \
    --base_url http://localhost:$PORT/v1 \
    --concurrency 128 \
    --batch_size 1024 \
    --max_tokens 4096 \
    --batch_mode async \
    --engine remote
```

`scripts/evaluate/eval_ftrl_vllm_binxing.sh` 也是 vLLM + ToolHop 评估模板，但示例模型为 `/ssd/model/Qwen3-235B-A22B-Thinking-2507`，端口为 `7899`，`tensor-parallel-size=2`，适合作为大模型 baseline 评估参考：

```bash
bash scripts/evaluate/eval_ftrl_vllm_binxing.sh
```

对应测试代码在 `evaluate/evaluation_toolhop.py`，binxing 脚本使用的是 `--series qwen --engine remote --batch_mode async` 路径，核心流程如下：

1. `main()` 读取 `data/toolhop/ToolHop.jsonl`，按 `start_id/end_id` 截取数据，并根据已有 `output_file` 跳过已完成样本。
2. 对 Qwen 系列加载 `AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)`，并通过 `get_parse_output("qwen")` 选择 `parse_qwen`。
3. `ChatVLLM(args)` 在 `engine=remote` 时创建 `AsyncLLMServer`，连接脚本里启动的 vLLM OpenAI-compatible endpoint。
4. 每个 batch 调用 `sample_process_async_batch()`；该函数按 `scenario` 构造初始 user prompt。
5. `Direct` 场景不传工具，且 `max_turns=1`；`Free` 和 `Mandatory` 场景把样本里的 `tools` 转成 OpenAI function schema。
6. 每个样本进入 `rollout_one_instance()`，最多循环 `max_turns` 轮。
7. 每轮通过 `engine.chat_one_async(messages, tools)` 生成回复；`AsyncLLMServer` 会用 tokenizer 把 `messages/tools` 拼成 chat template，并按 vLLM 返回的 `max_model_len` 做长度检查。
8. 回复文本通过 `parse_qwen()` 解析 `<tool_call>...</tool_call>`；如果存在 tool call，则调用 `prog_env.utils.utils.get_feedback()`，用样本内 `functions` 本地执行工具代码，并把 tool response 追加回 `messages`。
9. 如果模型不再输出 tool call，当前样本结束；最终用 `_answer_correct()` 检查 assistant final answer 或最后一个 tool response 是否包含 gold answer。
10. 结果逐行 append 到 `output_file`，每行包含 `messages`、`data_source`、`model`、`metrics.answer_correctness`。

因此 ToolHop/FTRL 风格评估不只是在远程模型上生成答案，而是复用了训练侧相同的“模型输出工具调用 -> 本地执行工具函数 -> tool response 继续对话”的多轮闭环。

### 3. BFCL 多轮函数调用评估

参考 `scripts/evaluate/eval_bfcl_vllm_binxing.sh`：

```bash
bash scripts/evaluate/eval_bfcl_vllm_binxing.sh
```

该脚本会：

- 在 `PORT=7897` 启动 vLLM；
- 设置 `REMOTE_OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1`；
- 设置 `REMOTE_OPENAI_TOKENIZER_PATH=$MODEL_PATH`；
- 使用 `bfcl generate` 生成结果；
- 使用 `bfcl evaluate` 计算分数；
- 输出到 `results/BFCL/Qwen3-4B/...`。

核心 BFCL 命令：

```bash
bfcl generate \
  --model Qwen/Qwen3-4B-FC \
  --test-category multi_turn \
  --skip-server-setup \
  --num-threads 32 \
  --result-dir $RESULT_DIR

bfcl evaluate \
  --model Qwen/Qwen3-4B-FC \
  --test-category multi_turn \
  --result-dir $RESULT_DIR \
  --score-dir $RESULT_DIR/scores
```

执行前需要检查：

- `MODEL_PATH` 是否指向已 merge 的 checkpoint；
- `RESULT_DIR` 是否与实验名一致；
- `CUDA_VISIBLE_DEVICES`、`TP_SIZE`、`DP_SIZE` 是否匹配当前机器；
- `conda activate BFCL` 环境是否可用；
- BFCL 脚本包含多个模型评估块，确认是否需要一次性全部运行。

BFCL 对应代码来自 vendored `evaluate/bfcl/berkeley-function-call-leaderboard/`，不是 `evaluate/evaluation_toolhop.py`。脚本里的 `bfcl generate` 和 `bfcl evaluate` 分别进入 BFCL CLI 的生成与打分流程：

1. `bfcl generate` 调用 BFCL 的 response generation 入口，读取 `--model` 和 `--test-category multi_turn`，根据 BFCL 自带数据集收集 involved test cases。
2. 因为脚本设置了 `--skip-server-setup`，BFCL 不自己启动 vLLM，而是使用脚本提前启动的服务；endpoint 来自 `LOCAL_SERVER_ENDPOINT`、`LOCAL_SERVER_PORT` 和 `REMOTE_OPENAI_BASE_URL`。
3. 生成阶段按 `--num-threads` 并发请求模型，并把结果写入 `$RESULT_DIR/<model_name>/...` 下的 BFCL 标准结果文件。
4. `bfcl evaluate` 调用 `eval_checker/eval_runner.py`，读取生成结果、按 `multi_turn` category 加载 checker，并输出 score 文件。
5. `eval_runner.py` 会汇总生成 `data_overall.csv`、`data_multi_turn.csv` 等 leaderboard CSV；脚本中 `--score-dir $RESULT_DIR/scores` 指定这些文件的落盘位置。

与 ToolHop 评估不同，BFCL 的工具调用执行、AST/exec 检查、多轮一致性打分都由 BFCL 自身 checker 完成；本仓库脚本主要负责 merge 后模型部署、环境变量配置、生成和评分命令串联。

### 4. 基础代码检查

文档/轻量代码检查仍可使用：

```bash
pytest verl/tests -q
ruff check verl
pre-commit run --all-files
python -m prog_env.advantage_est.mathtir_adv --bsz 8 --seq-len 64
```

这些检查只能验证基础代码和部分自定义 advantage 逻辑，不能替代上面的 merge + vLLM + benchmark 评估。

## 常见扩展方式

### 新增工具调用格式

在 `prog_env/agent_loop/tool_parser.py` 中新增 parser，并注册：

```python
@ToolParser.register("your_format")
class YourToolParser(ToolParser):
    ...
```

然后在训练脚本中设置：

```bash
actor_rollout_ref.rollout.multi_turn.format=your_format
```

### 新增 Agent Loop

在 `prog_env/agent_loop/` 中实现继承 `AgentLoopBase` 的类，并注册：

```python
@register("your_agent")
class YourAgentLoop(AgentLoopBase):
    ...
```

确保 `prog_env/agent_loop/__init__.py` 导入该类，然后通过样本字段 `agent_name` 或环境变量指定：

```bash
export AGENT_NAME="your_agent"
```

### 新增 Reward Function

在 `prog_env/reward_score/matchtir.py` 或新文件中实现函数，签名保持与现有 reward function 兼容：

```python
def compute_xxx(response, codes, unsolved_set, solve_rate, split, answer=None, gt_tool_call=None, tokenizer=None, valid_response_ids=None):
    ...
```

训练时指定：

```bash
custom_reward_function.path=$PROJECT_DIR/prog_env/reward_score/matchtir.py
custom_reward_function.name=compute_xxx
```

### 新增 Advantage Estimator

在 `prog_env/advantage_est/` 中使用 verl registry 注册：

```python
from verl.trainer.ppo.core_algos import register_adv_est

@register_adv_est("your_adv")
def compute_your_advantage(...):
    ...
```

确保 `prog_env/advantage_est/__init__.py` 导入该函数，然后配置：

```bash
algorithm.adv_estimator=your_adv
```

### 新增 Policy Loss

在 `prog_env/loss_cal/` 中使用 verl registry 注册：

```python
from verl.trainer.ppo.core_algos import register_policy_loss

@register_policy_loss("your_loss")
def compute_policy_loss_your_loss(...):
    ...
```

确保 `prog_env/loss_cal/__init__.py` 导入该函数，然后配置：

```bash
actor_rollout_ref.actor.policy_loss.loss_mode=your_loss
```

## 注意事项

- 当前脚本中的模型路径、conda 环境、cache 路径带有机器假设，迁移机器时需要先改脚本。
- `codes` 会通过 `exec` 在本地执行，训练数据必须可信；不要加载不可信 parquet。
- `tools` 字段需要是 tokenizer chat template 可接受的 tool schema 格式。
- 如果 prompt 长度过滤阶段报错，优先检查 `tools` 是否能被 `json.loads` 正确解析。
- 如果设置了 `AGENT_NAME`，它会覆盖样本内 `agent_name`。
- 如果使用 `hermes_prog`，模型需要在 `</think>` 后输出合法 `<tool_call>...</tool_call>`，否则该 step 不会被解析为工具调用。
