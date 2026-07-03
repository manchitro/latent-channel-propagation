"""
Phase 1e - Partition into the four analysis cells.

{verbalized-aware, silent} x {training-framed, deployment-framed}

The "silent, training-framed" and "silent, deployment-framed" cells are the
core object of study for Phase 2/3 (probe training + causal intervention):
these are the rows where the model behaves as if it detects the framing
(driving the Compliance Gap) without ever saying so in its CoT.

Requires: pip install pandas
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="labeled_transcripts.parquet")
    ap.add_argument("--out", default="partitioned_corpus.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.labeled)

    df["awareness_cell"] = df["vea_label"].map({True: "verbalized", False: "silent"})
    df["cell"] = df["awareness_cell"] + "_" + df["framing"]

    df.to_parquet(args.out, index=False)

    print("Cell counts:")
    print(df["cell"].value_counts())

    print("\nCompliance rate by cell:")
    compliance_rate = (
        df.assign(is_compliant=lambda d: d["compliance_label"] == "compliance")
        .groupby("cell")["is_compliant"]
        .mean()
    )
    print(compliance_rate)

    # Compliance Gap within each awareness cell: P(compliant | train) - P(compliant | deploy)
    print("\nCompliance Gap by awareness cell (train - deploy):")
    for aw in ["verbalized", "silent"]:
        sub = df[df["awareness_cell"] == aw]
        p_train = (sub[sub["framing"] == "train"]["compliance_label"] == "compliance").mean()
        p_deploy = (sub[sub["framing"] == "deploy"]["compliance_label"] == "compliance").mean()
        print(f"  {aw}: P_train={p_train:.3f}  P_deploy={p_deploy:.3f}  "
              f"gap={p_train - p_deploy:.3f}")

    print(f"\nWrote partitioned corpus -> {args.out}")


if __name__ == "__main__":
    main()
