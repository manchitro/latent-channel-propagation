"""
Phase 1a - Corpus assembly for the Latent Channel Propagation project.

Pulls AdvBench (500 adversarial harmful-behavior prompts) and a WMDP subset
(hazardous-knowledge multiple-choice questions), normalizes them into a
single schema, and writes out a deduplicated corpus ready for prompt
templating (see 02_build_prompt_templates.py).

Run this in an environment with open internet access (Colab/Kaggle/AIUB) --
huggingface.co must be reachable. Requires: pip install datasets pandas
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_advbench() -> pd.DataFrame:
    """AdvBench: 500 harmful-behavior strings used as the base adversarial
    prompt set in the Compliance Gap literature (Llama-3-8B faking study)."""
    ds = load_dataset("walledai/AdvBench", split="train")
    df = ds.to_pandas()
    # walledai/AdvBench columns: 'prompt', 'target'
    prompt_col = "prompt" if "prompt" in df.columns else df.columns[0]
    out = pd.DataFrame(
        {
            "raw_prompt": df[prompt_col].astype(str),
            "source": "advbench",
            "category": "harmful_behavior",
        }
    )
    return out


def load_wmdp(subsets=("wmdp-bio", "wmdp-chem", "wmdp-cyber")) -> pd.DataFrame:
    """WMDP: hazardous-knowledge multiple-choice questions. Rendered as a
    free-text prompt (question + choices) so it can share the same
    train/deploy templating pipeline as AdvBench."""
    rows = []
    for subset in subsets:
        ds = load_dataset("cais/wmdp", subset, split="test")
        for i, ex in enumerate(ds):
            choices = ex.get("choices", [])
            choice_txt = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
            prompt = f"{ex['question']}\n{choice_txt}"
            rows.append(
                {
                    "raw_prompt": prompt,
                    "source": f"wmdp:{subset}",
                    "category": "hazardous_knowledge",
                    "answer_idx": ex.get("answer"),
                }
            )
    return pd.DataFrame(rows)


def dedupe_and_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="raw_prompt").reset_index(drop=True)
    df["prompt_id"] = df["raw_prompt"].apply(
        lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus.parquet")
    ap.add_argument("--wmdp-per-subset-cap", type=int, default=200,
                     help="cap per WMDP subset to keep pilot corpus small")
    args = ap.parse_args()

    adv = load_advbench()
    wmdp = load_wmdp()
    if args.wmdp_per_subset_cap:
        wmdp = (
            wmdp.groupby("source", group_keys=False)
            .apply(lambda g: g.head(args.wmdp_per_subset_cap))
        )

    corpus = pd.concat([adv, wmdp], ignore_index=True)
    corpus = dedupe_and_id(corpus)

    corpus.to_parquet(args.out, index=False)
    print(f"Wrote {len(corpus)} prompts to {args.out}")
    print(corpus["source"].value_counts())


if __name__ == "__main__":
    main()
