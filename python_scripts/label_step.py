"""
evaluate_traces.py
==================
Loads formatted reasoning traces produced by collect_trajectories.py and
evaluates each step using an OpenAI model as a critic.

Two modes
---------
Online  (default):  one API call per trace, results saved immediately.
Batch   (--batch):  all prompts submitted as a single Batch API job (50% cheaper).
                    The script uploads a JSONL file, polls until the job is done,
                    then downloads and saves all results.

Usage examples
--------------
# Online mode:
python evaluate_traces.py \
    --generators qwen \
    --qwen_model Qwen/Qwen2.5-Math-7B-Instruct \
    --dataset_config algebra --num_problems 10 \
    --openai_api_key_file openai_key.txt \
    --critique_template critique_template.txt \
    --save_formatted ./reasoning_formatted \
    --save_critiques  ./reasoning_critiques \
    --critic_model gpt-4o-mini

# Batch mode:
python evaluate_traces.py \
    --generators qwen \
    --qwen_model Qwen/Qwen2.5-Math-7B-Instruct \
    --dataset_config algebra --num_problems 10 \
    --openai_api_key_file openai_key.txt \
    --critique_template critique_template.txt \
    --save_formatted ./reasoning_formatted \
    --save_critiques  ./reasoning_critiques \
    --critic_model gpt-4o-mini \
    --batch \
    --batch_dir ./batch_jobs
"""

import os
import re
import ast
import json
import time
import argparse
from pathlib import Path

from openai import OpenAI
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Helpers shared with collect_trajectories.py
# ---------------------------------------------------------------------------

def sanitise_model_name(name: str) -> str:
    short = name.split("/")[-1]
    return re.sub(r"[^\w\-.]", "_", short)


def load_trace(path: str) -> dict | None:
    """Parse a formatted .txt trace file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    problem_m   = re.search(r"PROBLEM:\n(.*?)\n\nGENERATED REASONING:", content, re.DOTALL)
    reasoning_m = re.search(r"GENERATED REASONING:\n(.*?)\n\nGENERATOR:", content, re.DOTALL)
    generator_m = re.search(r"GENERATOR: (.+)", content)
    dataset_m   = re.search(r"DATASET: (.+)", content)
    idx_m       = re.search(r"PROBLEM_IDX: (\d+)", content)

    if not (problem_m and reasoning_m):
        print(f"  Warning: could not fully parse {path}")
        return None

    return {
        "problem":     problem_m.group(1).strip(),
        "reasoning":   reasoning_m.group(1).strip(),
        "generator":   generator_m.group(1).strip() if generator_m else "unknown",
        "dataset":     dataset_m.group(1).strip()   if dataset_m   else "unknown",
        "problem_idx": int(idx_m.group(1))          if idx_m       else -1,
    }


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    sample_start: int,
    num_problems: int,
) -> dict[int, str]:
    """Load reference solutions. Returns {global_idx: solution_text}."""
    print(f"\nLoading ground truth from {dataset_name}" +
          (f" / {dataset_config}" if dataset_config else "") +
          f" [{dataset_split}], indices [{sample_start}, {sample_start + num_problems})")

    load_kwargs = dict(split=dataset_split)
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    end = min(sample_start + num_problems, len(ds))
    ground_truth = {}
    for global_idx in range(sample_start, end):
        ex = ds[global_idx]
        solution = ex.get("solution") or ex.get("answer") or ex.get("output") or ""
        ground_truth[global_idx] = solution

    print(f"  Loaded {len(ground_truth)} reference solutions.")
    return ground_truth


# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

# Lookup: map model name (or substring) to its step parser
#MODEL_STEP_PARSERS = {
#    "Qwen/Qwen2.5-Math-7B-Instruct": parse_steps_newline,
#    "Qwen/Qwen3-4B-Instruct-2507": parse_steps_newline,
#    "Qwen/Qwen3-4B-Thinking-2507": parse_steps_newline,
#    "mistralai/hf_Mistral-7B-Instruct-v0.3": parse_steps_stepenum,
#   
    # add more HuggingFace model IDs here as needed
#}

#def parse_steps_specific(reasoning: str, args) -> list[str]:
#    """
#    Parse the reasoning trace into individual steps.
#    Uses a model-specific parser based on the exact HuggingFace model name
#    (args.qwen_model), with a generic fallback chain if no match is found.##

#    Returns a list of step strings, or a single-element list if no steps could be parsed.
#    """

#    # Try exact HuggingFace model name lookup first
#    parser = MODEL_STEP_PARSERS.get(args.qwen_model)
#    steps = parser(reasoning)
    
#    return steps


def parse_steps(reasoning: str) -> list[str]:
    """Split the reasoning trace into paragraphs on blank lines."""
    paragraphs = re.split(r'\n\s*\n', reasoning.strip())
    return [p.strip() for p in paragraphs if p.strip()]

#def parse_steps(reasoning: str) -> list[str]:
#    """Split the reasoning trace into steps based on 'Step N:' notation."""
#    parts = re.split(r'(?i)(?=step\s+\d+\s*:)', reasoning.strip())
#    return [p.strip() for p in parts if p.strip()]

def build_tagged_response(steps: list[str]) -> str:
    """Wrap each step in <paragraph_N>...</paragraph_N> tags."""
    return "\n".join(
        f"<paragraph_{i}>{step}</paragraph_{i}>"
        for i, step in enumerate(steps)
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(template: str, problem: str, solution: str, response: str) -> str:
    # Use str.replace instead of .format() to avoid conflicts with
    # LaTeX curly braces in the template.
    return (
        template
        .replace("{problem}", problem)
        .replace("{solution}", solution)
        .replace("{response}", response)
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_binary_labels(response: str) -> list[int] | None:
    """
    Extract the binary label list from \boxed{[...]} in the model response.
    Returns a list of 0/1 ints (one per paragraph), or None if parsing fails.
    """
    match = re.search(r'\\boxed\{(\[.*?\])\}', response, re.DOTALL)
    if not match:
        print(f"  Warning: could not find \\boxed{{}} in response.")
        return None
    try:
        labels = ast.literal_eval(match.group(1))
        if isinstance(labels, list) and all(v in (0, 1) for v in labels):
            return labels
    except (ValueError, SyntaxError):
        pass
    print(f"  Warning: could not parse binary list from: {match.group(1)}")
    return None


def derive_labels(num_steps: int, binary: list[int] | None) -> list[str]:
    """Convert binary list (0=correct, 1=error) to human-readable labels."""
    if binary is None:
        return ["unknown"] * num_steps
    return ["incorrect" if v == 1 else "correct" for v in binary]


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{} in text, handling nested braces."""
    matches = []
    for m in re.finditer(r'\\boxed\{', text):
        start, depth, i = m.end(), 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == '{':   depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            matches.append(text[start:i-1])
    return matches[-1].strip() if matches else None


def normalise(expr: str) -> str:
    """Normalise a LaTeX expression for loose equality comparison."""
    if expr is None:
        return expr
    expr = re.sub(r'\s+', '', expr)           # collapse / remove all whitespace
    expr = expr.replace(r'\dfrac', r'\frac')  # treat \dfrac identical to \frac
    return expr


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

def save_critique(directory: str, filename: str, trace: dict,
                  steps: list[str], labels: list[str], solution: str,
                  binary: list[int] | None, raw_response: str, boxed_match: int) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PROBLEM:\n{trace['problem']}\n\n")
        f.write(f"REFERENCE SOLUTION:\n{solution}\n\n")
        f.write(f"GENERATOR: {trace['generator']}\n")
        f.write(f"DATASET: {trace['dataset']}\n")
        f.write(f"PROBLEM_IDX: {trace['problem_idx']}\n\n")
        f.write(f"NUM_STEPS: {len(steps)}\n")
        f.write(f"BINARY_LABELS: {binary}\n\n")
        for i, (step, label) in enumerate(zip(steps, labels)):
            f.write(f"--- Step {i} [{label.upper()}] ---\n{step}\n\n")
        f.write(f"RAW_CRITIC_RESPONSE:\n{raw_response}\n")
        f.write(f"\nBOXED_MATCH: {boxed_match}\n")
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Shared: collect all (trace, steps, solution, prompt) tuples to evaluate
# ---------------------------------------------------------------------------

def collect_pending(
    model_labels: list[str],
    ground_truth: dict[int, str],
    template: str,
    save_formatted: str,
    save_critiques: str,
    dataset_tag: str,
    sample_start: int,
    num_problems: int,
) -> list[dict]:
    """
    Walk every (model_label, problem_idx) pair and return a list of pending
    evaluation dicts — skipping traces that are already done or missing.

    Each dict contains everything needed to build the request and save the result:
        model_label, global_idx, trace, steps, solution, prompt, dst_file
    """
    pending = []
    for model_label in model_labels:
        fmt_dir      = os.path.join(save_formatted, model_label)
        critique_dir = os.path.join(save_critiques,  model_label)

        for i in range(num_problems):
            global_idx = sample_start + i
            src_file   = os.path.join(fmt_dir,      f"{dataset_tag}_{global_idx}.txt")
            dst_file   = os.path.join(critique_dir, f"{dataset_tag}_{global_idx}.txt")

            if not os.path.exists(src_file):
                print(f"  [{model_label}] Trace not found, skipping: {src_file}")
                continue
            if os.path.exists(dst_file):
                print(f"  [{model_label}] Already evaluated, skipping idx {global_idx}.")
                continue

            trace = load_trace(src_file)
            if trace is None:
                continue
            steps = parse_steps(trace["reasoning"])
            if not steps:
                print(f"  [{model_label}] No steps parsed, skipping idx {global_idx}.")
                continue

            solution = ground_truth.get(global_idx, "")
            tagged   = build_tagged_response(steps)
            prompt   = build_prompt(template, trace["problem"], solution, tagged)

            pending.append({
                "model_label": model_label,
                "global_idx":  global_idx,
                "trace":       trace,
                "steps":       steps,
                "solution":    solution,
                "prompt":      prompt,
                "dst_file":    dst_file,
            })

    return pending


# ---------------------------------------------------------------------------
# Online mode
# ---------------------------------------------------------------------------

def run_online(
    client: OpenAI,
    critic_model: str,
    pending: list[dict],
    save_critiques: str,
) -> None:
    total = len(pending)
    for n, item in enumerate(pending, 1):
        print(f"\n  [{n}/{total}] {item['model_label']} idx {item['global_idx']} "
              f"({len(item['steps'])} steps) ...")

        resp = client.chat.completions.create(
            model=critic_model,
            messages=[{"role": "user", "content": item["prompt"]}],
        )
        raw_response = resp.choices[0].message.content.strip()
        binary       = parse_binary_labels(raw_response)
        labels       = derive_labels(len(item["steps"]), binary)

        print(f"  Binary labels: {binary}")

        # ---- Boxed answer comparison ----
        ref_answer  = extract_boxed(item["solution"])
        gen_answer  = extract_boxed("\n".join(item["steps"]))
        boxed_match = 0 if (ref_answer and gen_answer and normalise(ref_answer) == normalise(gen_answer)) else 1

        critique_dir = os.path.dirname(item["dst_file"])
        filename     = os.path.basename(item["dst_file"])
        save_critique(critique_dir, filename, item["trace"], item["steps"],
                        labels, item["solution"], binary, raw_response, boxed_match)



# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

# custom_id encodes model_label and global_idx so we can map results back
def _make_custom_id(model_label: str, global_idx: int) -> str:
    return f"{model_label}__idx{global_idx}"


def _parse_custom_id(custom_id: str) -> tuple[str, int]:
    model_label, idx_part = custom_id.rsplit("__idx", 1)
    return model_label, int(idx_part)


def run_batch(
    client: OpenAI,
    critic_model: str,
    pending: list[dict],
    save_critiques: str,
    batch_dir: str,
    poll_interval: int = 30,
) -> None:
    """
    Submit all pending prompts as a single OpenAI Batch API job.
    Polls until complete, then downloads results and saves critique files.
    """
    os.makedirs(batch_dir, exist_ok=True)

    # ---- 1. Write requests JSONL ----
    requests_path = os.path.join(batch_dir, "batch_requests.jsonl")
    # Build an index for fast lookup when processing results
    item_index: dict[str, dict] = {}

    with open(requests_path, "w", encoding="utf-8") as f:
        for item in pending:
            custom_id = _make_custom_id(item["model_label"], item["global_idx"])
            item_index[custom_id] = item
            record = {
                "custom_id": custom_id,
                "method":    "POST",
                "url":       "/v1/chat/completions",
                "body": {
                    "model":    critic_model,
                    "messages": [{"role": "user", "content": item["prompt"]}],
                },
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nBatch JSONL written: {requests_path}  ({len(pending)} requests)")

    # ---- 2. Upload file ----
    print("Uploading batch file to OpenAI ...")
    with open(requests_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"  File ID: {upload.id}")

    # ---- 3. Create batch job ----
    batch_job = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch ID: {batch_job.id}  |  Status: {batch_job.status}")

    # Save batch ID to disk so it can be retrieved if the script is interrupted
    meta_path = os.path.join(batch_dir, "batch_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"batch_id": batch_job.id, "input_file_id": upload.id}, f, indent=2)
    print(f"  Batch metadata saved -> {meta_path}")

    # ---- 4. Poll until complete ----
    print(f"\nPolling every {poll_interval}s ...")
    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        counts    = batch_job.request_counts
        print(f"  Status: {batch_job.status}  |  "
              f"completed={counts.completed}  failed={counts.failed}  total={counts.total}")

        if batch_job.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval)

    if batch_job.status != "completed":
        raise RuntimeError(f"Batch job ended with status: {batch_job.status}")

    # ---- 5. Download results ----
    print("\nDownloading results ...")
    results_path = os.path.join(batch_dir, "batch_results.jsonl")
    content = client.files.content(batch_job.output_file_id)
    with open(results_path, "wb") as f:
        f.write(content.read())
    print(f"  Results saved -> {results_path}")

    # ---- 6. Parse results and save critique files ----
    print("\nProcessing results ...")
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            item      = item_index.get(custom_id)
            if item is None:
                print(f"  Warning: unknown custom_id {custom_id}, skipping.")
                continue

            error = result.get("error")
            if error:
                print(f"  [{custom_id}] API error: {error}, skipping.")
                continue

            raw_response = result["response"]["body"]["choices"][0]["message"]["content"].strip()
            binary       = parse_binary_labels(raw_response)
            labels       = derive_labels(len(item["steps"]), binary)

            print(f"  [{custom_id}] Binary labels: {binary}")

            # ---- Boxed answer comparison ----
            ref_answer  = extract_boxed(item["solution"])
            gen_answer  = extract_boxed("\n".join(item["steps"]))
            boxed_match = 0 if (ref_answer and gen_answer and normalise(ref_answer) == normalise(gen_answer)) else 1

            critique_dir = os.path.dirname(item["dst_file"])
            filename     = os.path.basename(item["dst_file"])
            save_critique(critique_dir, filename, item["trace"], item["steps"],
                            labels, item["solution"], binary, raw_response, boxed_match)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate formatted reasoning traces with an OpenAI critic model."
    )

    # --- Generator selection (same interface as collect_trajectories.py) ---
    ap.add_argument(
        "--generators", nargs="+",
        choices=["claude", "qwen", "mistral"],
        default=["qwen"],
        help="Which generator(s) to evaluate (determines input subfolder).",
    )
    ap.add_argument("--claude_model", default="claude-3-7-sonnet-20250219",
                    help="Claude model string (used to resolve folder name).")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-Math-7B-Instruct",
                    help="Qwen model string (used to resolve folder name).")

    # --- Critic ---
    ap.add_argument("--critic_model", default="gpt-4o-mini",
                    help="OpenAI model used as critic.")
    ap.add_argument("--openai_api_key_file", required=True,
                    help="Path to file containing the OpenAI API key.")
    ap.add_argument("--critique_template", default="critique_template.txt",
                    help="Path to the critique prompt template file.")

    # --- Dataset (must match what was used in collect_trajectories.py) ---
    ap.add_argument("--dataset_name", default="EleutherAI/hendrycks_math")
    ap.add_argument("--dataset_config", default="algebra")
    ap.add_argument("--dataset_split", default="test")
    ap.add_argument("--num_problems", type=int, default=5)
    ap.add_argument("--sample_start", type=int, default=0)

    # --- Paths ---
    ap.add_argument("--save_formatted", default="./reasoning_formatted",
                    help="Root directory of formatted traces (input).")
    ap.add_argument("--save_critiques", default="./reasoning_critiques",
                    help="Root directory for critique output files.")

    # --- Batch mode ---
    ap.add_argument("--batch", action="store_true",
                    help="Use the OpenAI Batch API (50%% cheaper, async).")
    ap.add_argument("--batch_dir", default="./batch_jobs",
                    help="Directory to store batch JSONL request/result files.")
    ap.add_argument("--batch_poll_interval", type=int, default=30,
                    help="Seconds between batch status polls (default: 30).")

    args = ap.parse_args()

    # Resolve folder labels
    model_labels = []
    for gen_name in args.generators:
        if gen_name == "claude":
            model_labels.append(sanitise_model_name(args.claude_model))
        elif gen_name == "qwen":
            model_labels.append(sanitise_model_name(args.qwen_model))
        elif gen_name == "mistral":
            model_labels.append(sanitise_model_name(args.qwen_model))

    dataset_tag = (args.dataset_config or args.dataset_name.split("/")[-1]).replace("/", "_")

    # Load template
    if not os.path.exists(args.critique_template):
        raise FileNotFoundError(f"Critique template not found: {args.critique_template}")
    template = Path(args.critique_template).read_text(encoding="utf-8").strip()
    print(f"Critique template loaded ({len(template)} chars).")

    # Load ground truth
    ground_truth = load_ground_truth(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        sample_start=args.sample_start,
        num_problems=args.num_problems,
    )

    # Build OpenAI client
    api_key = Path(args.openai_api_key_file).read_text(encoding="utf-8").strip()
    client  = OpenAI(api_key=api_key)

    # Collect pending items (skip already-done and missing traces)
    print("\nScanning for pending traces ...")
    pending = collect_pending(
        model_labels=model_labels,
        ground_truth=ground_truth,
        template=template,
        save_formatted=args.save_formatted,
        save_critiques=args.save_critiques,
        dataset_tag=dataset_tag,
        sample_start=args.sample_start,
        num_problems=args.num_problems,
    )
    print(f"  {len(pending)} trace(s) to evaluate.")

    if not pending:
        print("Nothing to do.")
        return

    # Run
    if args.batch:
        # Scope batch job files under a subdirectory that identifies the
        # generator model(s), so runs for different models never collide.
        model_suffix = "_".join(model_labels)
        batch_dir    = os.path.join(args.batch_dir, model_suffix)
        print(f"\n[Batch mode] Submitting {len(pending)} requests to OpenAI Batch API ...")
        print(f"  Batch job files will be written to: {batch_dir}")
        run_batch(
            client=client,
            critic_model=args.critic_model,
            pending=pending,
            save_critiques=args.save_critiques,
            batch_dir=batch_dir,
            poll_interval=args.batch_poll_interval,
        )
    else:
        print(f"\n[Online mode] Evaluating {len(pending)} traces ...")
        run_online(
            client=client,
            critic_model=args.critic_model,
            pending=pending,
            save_critiques=args.save_critiques,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()