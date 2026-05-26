---
name: lhtir-post-train-merge-test
description: Merge the latest completed LHTIR training checkpoint into a HuggingFace model and run post-training ToolHop and BFCL evaluations using the repository's existing `scripts/train/merge_binxing.sh` and `scripts/evaluate/*_binxing.sh` scripts. Use when the user asks to evaluate a recently trained checkpoint, compare post-train benchmark results, or automate the standard LHTIR merge plus ToolHop/BFCL workflow without inventing a new evaluation path.
---

# Lhtir Post Train Merge Test

## Overview

Use this skill to stay on the repository's standard post-train evaluation path: merge a saved `actor` checkpoint, start the repo's vLLM-backed benchmark scripts, and summarize ToolHop plus BFCL metrics.

## Workflow

1. Determine the target checkpoint.
   Prefer the latest completed `checkpoints/qwen3-4b_ftrl_multiturn/**/global_step_*/actor` by modification time unless the user names a specific experiment.
   Treat a completed checkpoint as an existing `global_step_xx/actor` directory, not an in-progress output directory.

2. Update the repository scripts instead of inventing new entrypoints.
   Edit `scripts/train/merge_binxing.sh`:
   set `ACTOR_DIR` to the chosen `global_step_xx/actor`
   set `TARGET_DIR` to `checkpoints/merged_checkpoints/<experiment>-<global_step_suffix>`

   Edit `scripts/evaluate/eval_toolhop_vllm_binxing.sh`:
   set `model` to the merged checkpoint directory
   set ToolHop output filenames to the current experiment prefix

   Edit `scripts/evaluate/eval_bfcl_vllm_binxing.sh`:
   keep only the current model block
   set `MODEL_PATH` to the merged checkpoint directory
   set `RESULT_DIR` to the current experiment result directory
   remove stale extra model blocks or stray characters that would break execution

3. Run the standard commands in order.
   `bash scripts/train/merge_binxing.sh`
   `bash scripts/evaluate/eval_toolhop_vllm_binxing.sh`
   `bash scripts/evaluate/eval_bfcl_vllm_binxing.sh`

4. Summarize results from files, not from memory.
   For ToolHop, read the three JSONL files under `results/ToolHop/Qwen3-4B/`.
   For BFCL, read `scores/data_overall.csv` and `scores/data_multi_turn.csv`.
   Use `scripts/summarize_results.py` in this skill to compute the final report.

## Constraints

- Reuse the repository's existing `*_binxing.sh` scripts.
- Do not create a separate merge or evaluation pipeline unless the user explicitly asks for a new one.
- Prefer non-destructive script edits that only change model paths, result paths, and stale extra blocks.
- If local port binding is sandbox-blocked, rerun the benchmark commands with escalation so vLLM can start.
- Report both result paths and the key log paths in the final answer.

## Expected Outputs

- Merge model directory:
  `checkpoints/merged_checkpoints/<experiment-name>`
- ToolHop outputs:
  `results/ToolHop/Qwen3-4B/<experiment>-Free-vllm-4096.jsonl`
  `results/ToolHop/Qwen3-4B/<experiment>-Direct-vllm-4096.jsonl`
  `results/ToolHop/Qwen3-4B/<experiment>-Mandatory-vllm-4096.jsonl`
- BFCL outputs:
  `results/BFCL/Qwen3-4B/<experiment>/scores/data_overall.csv`
  `results/BFCL/Qwen3-4B/<experiment>/scores/data_multi_turn.csv`

## Scripts

- `scripts/summarize_results.py`
  Run this after ToolHop and BFCL complete to print a compact summary for the current experiment.
