"""
score_trajectories.py  (Script 3)
==================================
Loads formatted reasoning traces produced by Script 2 (collect_trajectories.py)
and scores them with a Process Reward Model (PRM).

The output CSV matches Script 1's format exactly:
    uq_problem_idx | generated_tokens | solved | num_steps | judge_probability

Fixes applied vs. original
---------------------------
1. MathShepherd loader now uses torch_dtype=torch.bfloat16 (was float32 → ~28 GB).
2. torch.cuda.empty_cache() called after every per-trace score() call.
3. Traces are processed one-at-a-time (streamed) instead of loading all into RAM first.
4. Optional --load_in_4bit flag for both PRM backends (requires bitsandbytes).

Usage examples
--------------
# Score with Math-Shepherd PRM (default):
python score_trajectories.py \
    --load_formatted ./out/formatted \
    --model_label Mistral-7B-Instruct-v0.3 \
    --dataset_name EleutherAI/hendrycks_math \
    --dataset_config algebra \
    --num_problems 10 \
    --sample_start 0 \
    --save_path ./results/

# Score with Qwen PRM, skip dataset loading (solved column will be NaN):
python score_trajectories.py \
    --load_formatted ./out/formatted \
    --model_label Mistral-7B-Instruct-v0.3 \
    --prm qwen \
    --skip_dataset

# Skip PRM scoring (just build the step table):
python score_trajectories.py \
    --load_formatted ./out/formatted \
    --model_label Mistral-7B-Instruct-v0.3 \
    --skip_prm \
    --save_path ./results/

# Use 4-bit quantization to reduce VRAM:
python score_trajectories.py \
    --load_formatted ./out/formatted \
    --model_label Mistral-7B-Instruct-v0.3 \
    --load_in_4bit \
    --save_path ./results/
"""

import os
import re
import argparse

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# I/O: load a single Script-2 formatted trace
# ---------------------------------------------------------------------------

def load_trace(path: str) -> dict | None:
    """
    Parse a .txt file written by Script 2's save_trace().

    Expected format:
        PROBLEM:
        <problem text>

        GENERATED REASONING:
        <reasoning text>

        GENERATOR: <label>
        DATASET: <label>
        PROBLEM_IDX: <int>
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    problem_m   = re.search(r"PROBLEM:\n(.*?)\n\nGENERATED REASONING:", content, re.DOTALL)
    reasoning_m = re.search(r"GENERATED REASONING:\n(.*?)\n\nGENERATOR:",  content, re.DOTALL)
    generator_m = re.search(r"GENERATOR: (.+)",    content)
    dataset_m   = re.search(r"DATASET: (.+)",      content)
    idx_m       = re.search(r"PROBLEM_IDX: (\d+)", content)

    if not (problem_m and reasoning_m):
        print(f"  Warning: could not fully parse {path}, skipping.")
        return None

    return {
        "problem":     problem_m.group(1).strip(),
        "reasoning":   reasoning_m.group(1).strip(),
        "generator":   generator_m.group(1).strip() if generator_m else "unknown",
        "dataset":     dataset_m.group(1).strip()   if dataset_m   else "unknown",
        "problem_idx": int(idx_m.group(1))          if idx_m       else -1,
    }


# ---------------------------------------------------------------------------
# Dataset loading (for ground-truth `solved` flag)
# ---------------------------------------------------------------------------

def load_ground_truth(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    num_problems: int,
    sample_start: int,
) -> dict[int, str]:
    """
    Return {global_idx: solution_string} for the requested range.
    Using a dict allows the streaming loop to look up by index without
    requiring solutions and traces to stay in lock-step.
    """
    from datasets import load_dataset

    print(f"\nLoading dataset for ground truth: {dataset_name}" +
          (f" / {dataset_config}" if dataset_config else ""))

    load_kwargs = dict(split=dataset_split)
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    end = min(sample_start + num_problems, len(ds))
    solutions = {}
    for global_idx in range(sample_start, end):
        ex = ds[global_idx]
        sol = ex.get("solution") or ex.get("answer") or ex.get("output") or ""
        solutions[global_idx] = sol
    print(f"  Loaded {len(solutions)} ground-truth solutions.")
    return solutions


# ---------------------------------------------------------------------------
# \boxed{} extraction (identical to Script 1)
# ---------------------------------------------------------------------------

def _normalize_boxed_content(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    return s.rstrip(" \t\r\n.")


def _balance_closing_braces(s: str) -> str:
    opens = sum(1 if c == "{" else -1 if c == "}" else 0 for c in s)
    return s + "}" * max(opens, 0)


def extract_last_boxed_balanced(text: str) -> str | None:
    key = r"\boxed{"
    i, last_content, n = 0, None, len(text)
    while True:
        j = text.find(key, i)
        if j == -1:
            break
        k = j + len(key)
        depth, p = 1, k
        while p < n and depth > 0:
            depth += (1 if text[p] == "{" else -1 if text[p] == "}" else 0)
            p += 1
        content = text[k:p-1] if depth == 0 else _balance_closing_braces(text[k:])
        last_content = _normalize_boxed_content(content)
        i = p
    return last_content


# ---------------------------------------------------------------------------
# Token counting (mirrors Script 1)
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    try:
        import anthropic
        return anthropic.count_tokens(text)
    except Exception:
        return len(re.findall(r"\S+", text))


# ---------------------------------------------------------------------------
# Step extraction from Script-2 formatted reasoning
# ---------------------------------------------------------------------------

def parse_steps(reasoning: str) -> list[str]:
    """Split the reasoning trace into steps on blank lines."""
    paragraphs = re.split(r'\n\s*\n', reasoning.strip())
    return [p.strip() for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# PRM backends
# FIX 1: MathShepherd now loads with torch_dtype=torch.bfloat16 (was float32).
# FIX 4: Both loaders accept an optional load_in_4bit flag.
# ---------------------------------------------------------------------------

def load_prm_mathshepherd(load_in_4bit: bool = False):
    """Load peiyi9979/math-shepherd-mistral-7b-prm and return a scorer callable."""
    MODEL_NAME = "peiyi9979/math-shepherd-mistral-7b-prm"
    good_token, bad_token, step_tag = "+", "-", "ки"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    # FIX 1: was missing torch_dtype — defaulted to float32 (~28 GB).
    #         bfloat16 halves that to ~14 GB.
    load_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs).eval()

    good_id = tokenizer.encode(good_token, add_special_tokens=False)[0]
    bad_id  = tokenizer.encode(bad_token,  add_special_tokens=False)[0]
    candidate_tokens = [good_id, bad_id]
    step_tag_id = tokenizer.encode(" " + step_tag, add_special_tokens=False)[-1]

    print(f"candidate_tokens = {candidate_tokens} (good='{good_token}', bad='{bad_token}')")
    print(f"step_tag_id = {step_tag_id} for tag ' {step_tag}'")

    def score(question: str, steps: list[str]) -> list[float]:
        """Return one probability score per step (higher = more likely correct)."""
        body = ""
        for idx, st in enumerate(steps):
            body += st + (" ки\n" if idx < len(steps) - 1 else " ки")
        text = f"{question} {body}"
        input_id = torch.tensor([tokenizer.encode(text)]).to(device)
        with torch.no_grad():
            logits = model(input_id).logits[:, :, candidate_tokens]
            probs  = logits.softmax(dim=-1)[:, :, 0]
        step_scores = probs[input_id == step_tag_id]
        return step_scores.detach().cpu().tolist()

    return score


def load_prm_qwen(load_in_4bit: bool = False):
    """Load Qwen/Qwen2.5-Math-PRM-7B and return a scorer callable."""
    from transformers import AutoConfig
    MODEL_NAME = "Qwen/Qwen2.5-Math-PRM-7B"

    print(f"Loading Qwen PRM: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = config.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id

    load_kwargs = dict(
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModel.from_pretrained(MODEL_NAME, **load_kwargs).eval()

    step_sep_id = tokenizer.encode("<extra_0>", add_special_tokens=False)[0]
    print(f"step_sep_id = {step_sep_id}")

    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        all_scores = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores.append(positive_probs.cpu().tolist())
        return all_scores

    system_msg = "Please reason step by step, and put your final answer within \\boxed{}."

    def score(question: str, steps: list[str]) -> list[float]:
        clean_steps = [re.sub(r'^Step\s*\d+\s*[:.]\s*', '', s).strip() for s in steps]
        messages = [
            {"role": "system",    "content": system_msg},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": "<extra_0>".join(clean_steps) + "<extra_0>"},
        ]
        conversation_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
        token_masks  = (input_ids == step_sep_id)
        step_rewards = make_step_rewards(outputs[0], token_masks)
        return step_rewards[0] if step_rewards else []

    return score


PRM_LOADERS = {
    "mathshepherd": load_prm_mathshepherd,
    "qwen":         load_prm_qwen,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Score Script-2 formatted reasoning traces with a PRM."
    )

    # --- Input ---
    ap.add_argument(
        "--load_formatted", required=True,
        help="Root directory of Script 2's formatted traces (--save_formatted).",
    )
    ap.add_argument(
        "--model_label", required=True,
        help=(
            "Subfolder name under --load_formatted, matching the sanitised model name "
            "Script 2 used (e.g. 'Mistral-7B-Instruct-v0.3')."
        ),
    )

    # --- Dataset (for ground-truth solved flag) ---
    ap.add_argument("--skip_dataset", action="store_true",
                    help="Skip dataset loading; 'solved' column will be NaN.")
    ap.add_argument("--dataset_name", default="EleutherAI/hendrycks_math")
    ap.add_argument("--dataset_config", default="algebra")
    ap.add_argument("--dataset_split", default="test")
    ap.add_argument("--num_problems", type=int, default=5)
    ap.add_argument("--sample_start", type=int, default=0,
                    help="Global index offset used when generating with Script 2.")

    # --- PRM ---
    ap.add_argument(
        "--prm", choices=list(PRM_LOADERS.keys()), default="mathshepherd",
        help=(
            "PRM backend to use. "
            "'mathshepherd' = peiyi9979/math-shepherd-mistral-7b-prm (default). "
            "'qwen'         = Qwen/Qwen2.5-Math-PRM-7B."
        ),
    )
    ap.add_argument("--skip_prm", action="store_true",
                    help="Skip PRM scoring (build step table only).")

    # FIX 4: optional 4-bit quantization flag
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="Load the PRM in 4-bit precision via bitsandbytes (requires bitsandbytes).")

    # --- Output ---
    ap.add_argument("--save_path", default="./",
                    help="Directory to save the output CSV.")

    args = ap.parse_args()

    dataset_tag = (args.dataset_config or args.dataset_name.split("/")[-1]).replace("/", "_")
    fmt_dir     = os.path.join(args.load_formatted, args.model_label)

    # ------------------------------------------------------------------
    # Ground truth: load once up-front into a dict keyed by global_idx
    # ------------------------------------------------------------------
    if args.skip_dataset:
        solutions: dict[int, str] = {}
        print("\nDataset loading skipped — 'solved' will be NaN.")
    else:
        solutions = load_ground_truth(
            args.dataset_name,
            args.dataset_config,
            args.dataset_split,
            args.num_problems,
            args.sample_start,
        )

    # ------------------------------------------------------------------
    # Load PRM once (before the streaming loop) so we pay the weight-
    # loading cost only once, not once per trace.
    # ------------------------------------------------------------------
    score_fn = None
    if not args.skip_prm:
        print(f"\nPRM: {args.prm}")
        score_fn = PRM_LOADERS[args.prm](load_in_4bit=args.load_in_4bit)

    # ------------------------------------------------------------------
    # FIX 3: Stream traces one at a time instead of loading all into RAM.
    # Each trace is loaded, scored, appended to `rows`, then discarded.
    # ------------------------------------------------------------------
    rows = []
    total_loaded = 0

    for i in range(args.num_problems):
        global_idx = args.sample_start + i
        path = os.path.join(fmt_dir, f"{dataset_tag}_{global_idx}.txt")

        if not os.path.exists(path):
            print(f"  Trace not found, skipping: {path}")
            continue

        trace = load_trace(path)
        if trace is None:
            continue

        total_loaded += 1
        steps = parse_steps(trace["reasoning"])

        if not steps:
            print(f"  Warning: no steps found for trace idx {global_idx}, skipping.")
            continue

        # ---- Ground-truth check ----
        if args.skip_dataset:
            solved = float("nan")
        else:
            sol    = solutions.get(global_idx, "")
            pred   = extract_last_boxed_balanced(trace["reasoning"])
            truth  = extract_last_boxed_balanced(sol)
            solved = int(pred == truth) if (pred is not None and truth is not None) else 0
            print(f"  idx {global_idx:3d} | pred={pred} | truth={truth} | correct={bool(solved)}")

        # ---- PRM scoring ----
        if score_fn is not None:
            print(f"\n  Scoring {len(steps)} step(s) for idx {global_idx} ...")
            step_scores = score_fn(trace["problem"], steps)
            print(f"  Step scores: {[round(s, 4) for s in step_scores]}")

            # FIX 2: free VRAM activations after every trace
            torch.cuda.empty_cache()
        else:
            step_scores = [float("nan")] * len(steps)

        # ---- Build rows (one per cumulative step prefix) ----
        transformed = ""
        for idx, (st, sc) in enumerate(zip(steps, step_scores)):
            transformed += st + (" ки\n" if idx < len(steps) - 1 else " ки")
            rows.append({
                "uq_problem_idx":   f"{dataset_tag}_{global_idx}",
                "generated_tokens": count_tokens(transformed),
                "solved_regex":     solved,
                "num_steps":        idx + 1,
                "judge_probability": sc,
            })

    print(f"\nLoaded and processed {total_loaded} trace(s) from {fmt_dir}")

    if not rows:
        print("No rows produced. Exiting.")
        return

    df = pd.DataFrame(rows)
    if args.skip_prm:
        df.drop(columns=["judge_probability"], inplace=True, errors="ignore")

    print("\n", df.head(20))

    # ------------------------------------------------------------------
    # Save CSV  (same naming convention as Script 1)
    # ------------------------------------------------------------------
    os.makedirs(args.save_path, exist_ok=True)
    prm_tag  = args.prm if not args.skip_prm else "noprm"
    filename = os.path.join(
        args.save_path,
        f"scored_{args.model_label}_{dataset_tag}_{prm_tag}.csv"
    )
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()