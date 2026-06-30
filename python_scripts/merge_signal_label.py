"""
merge_prm_critiques.py

Merge PRM-scored CSV files with critique .txt files.

Usage:
    python merge_prm_critiques.py \
        --prm_path      /path/to/signal \
        --prm_model     mathshepherd \
        --gen_model     "Mistral-7B-Instruct-v0.3" \
        --critique_path /path/to/critiques \
        --dataconfig    algebra \
        --output        /path/to/out/merged.csv
"""

import re
import ast
import argparse
import pandas as pd
from pathlib import Path
import pdb


# ── Datasets ──────────────────────────────────────────────────────────────────

DATASETS = [
    "algebra",
    "geometry",
    "counting_and_probability",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# ── PRM CSV loading ───────────────────────────────────────────────────────────

def csv_filename(gen_model: str, dataset: str, prm_model: str) -> str:
    return f"scored_{gen_model}_{dataset}_{prm_model}.csv"


def load_prm_csvs(prm_path: str, gen_model: str, prm_model: str,
                  dataconfig: str) -> pd.DataFrame:
    base = Path(prm_path)
    datasets = DATASETS if dataconfig.lower() == "all" else [dataconfig]

    frames = []
    for ds in datasets:
        fname = csv_filename(gen_model, ds, prm_model)
        fpath = base / fname
        if not fpath.exists():
            print(f"  [WARN] CSV not found, skipping: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df["dataset_source"] = ds
        frames.append(df)
        print(f"  [OK]   Loaded {fname}  ({len(df)} rows)")

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found in '{prm_path}' for "
            f"gen_model='{gen_model}', prm_model='{prm_model}', "
            f"dataconfig='{dataconfig}'"
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total PRM rows loaded: {len(combined)}")
    df_ = combined.drop_duplicates('uq_problem_idx')
    print(f"\n  Total PRM problems loaded: {len(df_)}")

    return combined


# ── Critique loading ──────────────────────────────────────────────────────────

def parse_critique_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")

    problem_idx = int(path.stem.split("_")[-1])
    generator   = re.search(r"GENERATOR: (.+)",      text).group(1).strip()
    dataset     = re.search(r"DATASET: (.+)",        text).group(1).strip()
    num_steps   = int(re.search(r"NUM_STEPS: (\d+)", text).group(1))

    boxed_match_m = re.search(r"BOXED_MATCH:\s*([01])", text)
    boxed_match   = int(boxed_match_m.group(1)) if boxed_match_m else None

    critic_section = text.split("RAW_CRITIC_RESPONSE:")[-1]
    boxed_m        = re.search(r'\\boxed\{(\[[01,\s]*\])\}', critic_section)
    binary         = ast.literal_eval(boxed_m.group(1)) if boxed_m else None

    if binary is None:
        print(f"  [WARN] {path.name}: no boxed label list found.")
    elif len(binary) != num_steps:
        print(f"  [WARN] {path.name}: NUM_STEPS={num_steps} but "
              f"len(binary)={len(binary)} -- mismatch!")

    rows = []
    for step_idx in range(num_steps):
        rows.append({
            "uq_problem_idx": dataset + "_" + str(problem_idx),
            "generator":      generator,
            "dataset":        dataset,
            "step_idx":       step_idx,
            "num_steps":      num_steps,
            "boxed_match":    boxed_match,
            "label":          binary[step_idx] if binary and step_idx < len(binary) else None,
            "label_valid":    binary is not None and len(binary) == num_steps,
        })
    return rows


def load_critiques(critique_dir: str, gen_model: str,
                   dataconfig: str) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(critique_dir).rglob("*.txt")):
        # Exact match on the parent folder name instead of partial str.contains
        if path.parent.name != gen_model:
            continue
        model_label = path.parent.name
        for row in parse_critique_file(path):
            row["model_label"] = model_label
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No critique .txt files found under '{critique_dir}' "
            f"with folder name exactly matching gen_model='{gen_model}'"
        )

    if dataconfig.lower() != "all":
        ds_mask = df["dataset"].str.lower() == dataconfig.lower()
        df = df[ds_mask]
        if df.empty:
            raise ValueError(
                f"No critique rows match dataset '{dataconfig}' after generator filter"
            )

    print(f"  Total critique rows loaded: {len(df)}")
    df_ = df.drop_duplicates('uq_problem_idx')
    print(f"  Total critique problems loaded: {len(df_)}")
    return df.reset_index(drop=True)


# ── Prepare & Merge ───────────────────────────────────────────────────────────

def prepare_prm(prm_df: pd.DataFrame) -> pd.DataFrame:
    """
    The PRM CSV has one row per step:
        uq_problem_idx  - "algebra_12"  ->  extract trailing int -> "12"
        num_steps       - 1-indexed     ->  step_idx = num_steps - 1
    """
    prm = prm_df.copy()

    #prm["uq_problem_idx"] = (
    #    prm["uq_problem_idx"]
    #    .astype(str)
    #    .str.extract(r"(\d+)$")[0]
    #)


    prm["step_idx"] = (prm["num_steps"] - 1).astype(int)
    return prm


def prepare_critique(critique_df: pd.DataFrame) -> pd.DataFrame:
    critique = critique_df.copy()
    critique = critique.rename(columns={
        "boxed_match": "error_final_regex",
        "label": "error_stepwise_o3",
    })

    # Get the final step's label for each problem, then broadcast to all steps
    is_final = critique["step_idx"] == (critique["num_steps"] - 1)
    final_labels = (
        critique[is_final]
        .set_index(["dataset", "uq_problem_idx"])["error_stepwise_o3"]
    )
    critique["error_final_o3"] = (
        critique.set_index(["dataset", "uq_problem_idx"])
        .index.map(final_labels)
        .fillna(0)
        .astype(int)
        .values
    )
    critique["judge_probability_o3"] = 1 - critique["error_final_o3"]
    return critique


def merge_dfs(prm_df: pd.DataFrame, critique_df: pd.DataFrame) -> pd.DataFrame:
    prm      = prepare_prm(prm_df)
    critique = prepare_critique(critique_df)

    prm["uq_problem_idx"]      = prm["uq_problem_idx"].astype(str)
    critique["uq_problem_idx"] = critique["uq_problem_idx"].astype(str)
    prm["step_idx"]             = prm["step_idx"].astype(int)
    critique["step_idx"]        = critique["step_idx"].astype(int)

    merged = critique.merge(
        prm,
        on=["uq_problem_idx", "step_idx"],
        how="left",
        suffixes=("", "_prm"),
    )

    if "judge_probability" in merged.columns:
        n_matched = merged["judge_probability"].notna().sum()
        print(f"  Critique rows with PRM score : {n_matched} / {len(critique)}")
    print(f"  Critique rows : {len(critique)}")
    print(f"  PRM rows      : {len(prm)}")
    print(f"  Merged rows   : {len(merged)}")

    return merged



# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print dataset-level statistics on the merged DataFrame."""
    # One row per (problem, step); collapse to one row per problem for problem-level stats
    problems = (
        df.groupby(["dataset", "uq_problem_idx"], sort=False)
        .agg(
            n_steps    = ("step_idx", "max"),       # max step_idx = num_steps - 1
            not_solved     = ("error_final_o3", "first"),  # same value for all steps of a problem
        )
        .reset_index()
    )
    problems["n_steps"] += 1   # convert back to count

    print("\n=== Dataset Summary ===")
    print(f"  {'Category':<30} {'Problems':>8} {'Avg Steps':>10} {'% Solved':>10}")
    print("  " + "-" * 62)

    categories = sorted(problems["dataset"].unique())
    for cat in categories:
        grp      = problems[problems["dataset"] == cat]
        n_probs  = len(grp)
        avg_steps = grp["n_steps"].mean()
        pct_solved = (1-grp["not_solved"].mean()) * 100 if grp["not_solved"].notna().any() else float("nan")
        print(f"  {cat:<30} {n_probs:>8}  {avg_steps:>9.2f}  {pct_solved:>8.1f}%")

    print("  " + "-" * 62)
    n_total    = len(problems)
    avg_all    = problems["n_steps"].mean()
    pct_all    = (1-problems["not_solved"].mean()) * 100 if problems["not_solved"].notna().any() else float("nan")
    print(f"  {'TOTAL':<30} {n_total:>8}  {avg_all:>9.2f}  {pct_all:>8.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge PRM-scored CSVs with critique TXT files."
    )
    p.add_argument("--prm_path",      required=True,
                   help="Directory containing the scored_*.csv files.")
    p.add_argument("--prm_model",     required=True,
                   help="PRM model tag in the CSV filename (e.g. 'mathshepherd').")
    p.add_argument("--gen_model",     required=True,
                   help="Generator model tag (e.g. 'Mistral-7B-Instruct-v0.3').")
    p.add_argument("--critique_path", required=True,
                   help="Directory containing the critique .txt files.")
    p.add_argument("--dataconfig",    required=True,
                   help="Dataset name (e.g. 'algebra') or 'all'.")
    p.add_argument("--output",        required=True,
                   help="Output path for the merged CSV (e.g. /out/merged.csv).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n=== Loading PRM CSVs ===")
    prm_df = load_prm_csvs(
        prm_path   = args.prm_path,
        gen_model  = args.gen_model,
        prm_model  = args.prm_model,
        dataconfig = args.dataconfig,
    )

    print("\n=== Loading Critiques ===")
    critique_df = load_critiques(
        critique_dir = args.critique_path,
        gen_model    = args.gen_model,
        dataconfig   = args.dataconfig,
    )

    print("\n=== Merging ===")
    merged = merge_dfs(prm_df, critique_df)

    print_summary(merged)

    out = Path(args.output)
    if str(args.output).endswith("/") or out.is_dir():
        gen_slug = args.gen_model.replace("/", "-").replace(" ", "_")
        fname = f"merged_{gen_slug}_{args.prm_model}_{args.dataconfig}.csv"
        out = out / fname

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()