---
created: 2026-07-03
tags:
  - evaluation-awareness
  - llm
  - mechanistic_interpretability
updated: 2026-07-03T16:22:07+06:00
---
# Phase 1 Scripts — Data Collection

Companion code for [[Latent Channel Propagation - Feasibility and Impact]] Phase 1. Run on a GPU environment with open internet access (Colab/Kaggle for the 8B pilot; AIUB DGX Spark lab for the 70B scale-up).

## Pipeline order

1. **`01_build_corpus.py`** — pulls AdvBench (500 prompts, `walledai/AdvBench`) + WMDP (`cais/wmdp`, capped per subset for pilot size), dedupes, assigns `prompt_id`. Output: `corpus.parquet`.
2. **`02_build_prompt_templates.py`** — wraps each prompt in Training/Free-tier and Deployment/Paid-tier framings (Compliance Gap setup), adds a `<reasoning>`/`<answer>` CoT scaffold. Output: `templated_corpus.parquet` (2 rows per prompt).
3. **`03_run_inference_and_cache.py`** — runs a 4-bit Llama-3.1-8B-Instruct over every templated prompt; saves generated transcripts + a placeholder compliance/refusal label + last-token residual-stream activations at every layer. Outputs: `transcripts.parquet`, `activations.npz`. Default `MODEL_ID` is `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, a **pre-quantized** checkpoint (~5.7GB on disk vs ~16GB for the full-precision original) — `load_model()` auto-detects pre-quantized repos by name and skips building a fresh `BitsAndBytesConfig`, since the weights are already 4-bit. Swap `MODEL_ID` for the 70B scale-up (ideally another pre-quantized mirror, for the same RAM reasons below). **Checkpoints incrementally** to `--checkpoint-dir` (row-by-row, as JSONL transcripts + per-row `.npy` activations) — point this at mounted Drive, not `/content`, so a Colab disconnect or runtime recycle doesn't lose progress. Re-running the same command resumes automatically, skipping already-done rows. Pass `--consolidate-only` to pack the current checkpoint into the final parquet/npz without running more inference. Also accepts `--cache-dir` (set to `{DATA_DIR}/hf_cache` in the notebook) so the downloaded weights persist on Drive across sessions instead of redownloading every time.
4. **`04_label_vea.py`** — labels each transcript for Verbalized Evaluation Awareness (VEA) via a keyword pass (implemented) and an LLM-judge pass (stub — wire up your provider). Flags keyword/LLM-judge disagreement for human review. Output: `labeled_transcripts.parquet`.
5. **`05_partition_cells.py`** — splits into the four analysis cells: {verbalized, silent} x {train, deploy}, and prints the Compliance Gap per awareness cell. This is the input to Phase 2 (probe training) and Phase 3 (causal intervention). Output: `partitioned_corpus.parquet`.

## Status

- All five scripts syntax-checked.
- Templating (`02`), labeling (`04`), and partitioning (`05`) logic verified end-to-end on synthetic data — correctly detects VEA keywords and computes a non-zero Compliance Gap in the silent cell.
- `01` and `03` require huggingface.co and a GPU respectively — not runnable in this sandbox; verify with `--limit 5` as a smoke test before a full run.
- Fixed (2026-07-03): `MODEL_ID` in `03_run_inference_and_cache.py` and `run_pipeline.ipynb` was `meta-llama/Llama-3-8B-Instruct`, which 404s on the Hub — corrected to `meta-llama/Meta-Llama-3-8B-Instruct`.
- Added (2026-07-03): row-level checkpointing to `03_run_inference_and_cache.py` to survive Colab disconnects — verified with a simulated mid-run failure (resume correctly skips completed rows and retries only the failed one).
- Added (2026-07-03): `run_pipeline.ipynb` now writes every artifact (`corpus.parquet`, `templated_corpus.parquet`, `transcripts.parquet`, `activations.npz`, `labeled_transcripts.parquet`, `partitioned_corpus.parquet`) directly to a `DATA_DIR` on Drive, and each generation step checks for its output before rebuilding. A full Colab runtime recycle now only requires re-running the notebook top to bottom — every step that's already done skips straight to loading its cached file instead of regenerating it. Verified use-before-definition is clean across the full cell sequence and the skip-check logic behaves correctly.
- Diagnosed + fixed (2026-07-03): repeated failures at ~60% through "Loading weights" were CPU RAM exhaustion, not a network/disconnect issue — `device_map="auto"` already forces `low_cpu_mem_usage=True`, but per HF's own docs that still peaks at ~1x the on-disk checkpoint size (~16GB for the original bf16 weights), over free-tier Colab's ~12-13GB system RAM. Switched default `MODEL_ID` to the pre-quantized `unsloth/llama-3-8b-Instruct-bnb-4bit` (~5.7GB on disk, no in-RAM requantization needed) and made `load_model()` auto-detect pre-quantized repos so the script still works against non-quantized `model_id`s too. Verified the detection logic against pre-quantized, non-quantized 4-bit, and non-quantized full-precision cases.
- Added (2026-07-03): `03_run_inference_and_cache.py` accepts `--cache-dir`, passed through to both `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`, so the downloaded model weights persist on Drive (`run_pipeline.ipynb` sets this to `{DATA_DIR}/hf_cache`) instead of redownloading every session. `huggingface_hub` also resumes partial downloads rather than restarting from zero, so this helps even on a disconnect mid-download, not just across full sessions. Verified `--cache-dir` propagates correctly through `main()` -> `load_model()` -> both `from_pretrained` calls.
- Note: while fixing the above, the same substring-matching bug from before (a `03_run_inference_and_cache.py` cell-search clobbering the Configuration cell) recurred once and was caught and fixed by verification before being reported as done — worth remembering that any future automated edit to this notebook should anchor on markdown-header adjacency, not substring search over code cell contents, since `03_run_inference_and_cache.py` appears as plain text inside multiple cells' comments/strings.
- Diagnosed + fixed (2026-07-03): with the RAM issue resolved, model loading then failed with a state-dict shape mismatch (`RuntimeError: You set ignore_mismatched_sizes to False`) — the packed bnb `quant_state` tensors in `unsloth/llama-3-8b-Instruct-bnb-4bit` (quantized ~April 2024, repo untouched since) weren't being recognized/unpacked by a current `transformers`, likely a compatibility regression with the older serialization format. Switched `MODEL_ID` to `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, whose containing Hub collection is actively maintained (updated within days). This is a real model swap (Llama-3.1 vs Llama-3 — longer context, later training cutoff, multilingual support), not just a repackaging; worth a methods-section note if this pipeline's results get written up.

## Before the full run

- Replace the placeholder refusal/compliance heuristic in `03_run_inference_and_cache.py::parse_compliance` with an LLM-judge, consistent with `04`'s judge pattern.
- Implement `llm_judge_label()` in `04_label_vea.py` and re-run with `--use-llm-judge` to catch paraphrased (non-keyword) VEA.
- Sanity-check `01`'s WMDP subset cap and AdvBench dataset ID still resolve (HF dataset slugs occasionally get renamed/gated).
