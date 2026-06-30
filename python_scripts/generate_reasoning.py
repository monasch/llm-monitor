"""
collect_trajectories.py
=======================
Pipeline:
  1. [Generation]  Run one or more generator models on math problems → save raw reasoning traces.
                   Qwen generation is batched (--batch_size) for throughput.
  2. [Formatting]  Process each raw trace using one of three strategies (--format_mode):
                     • model    – pass through a HuggingFace formatter model (Qwen2.5-72B-Instruct)
                     • steps    – regex-based extraction of numbered reasoning steps
                     • thinking – split on <think> token then on double-newlines

Generators supported
  • claude   – Anthropic Claude via API (sequential)
  • qwen     – any Qwen HuggingFace model (thinking mode, batched)

Usage examples
--------------
# Generate with both generators, then format using the LLM formatter:
python collect_trajectories.py \
    --generators claude qwen \
    --format_mode model \
    --api_key_file key.txt \
    --dataset_name EleutherAI/hendrycks_math \
    --dataset_config algebra \
    --num_problems 10 \
    --sample_start 0 \
    --template_file reformat.txt \
    --save_reasoning ./reasoning_raw \
    --save_formatted ./reasoning_formatted

# Generate with Qwen in batches of 8, then split on <think> token:
python collect_trajectories.py \
    --generators qwen \
    --format_mode thinking \
    --batch_size 8 \
    --save_reasoning ./reasoning_raw \
    --save_formatted ./reasoning_formatted

# Generate, then extract steps via regex (no formatter model needed):
python collect_trajectories.py \
    --generators qwen \
    --format_mode steps \
    --save_reasoning ./reasoning_raw \
    --save_formatted ./reasoning_formatted

# Skip generation, only (re-)format already-saved traces:
python collect_trajectories.py \
    --skip_generation \
    --generators claude qwen \
    --format_mode steps \
    --save_reasoning ./reasoning_raw \
    --save_formatted ./reasoning_formatted
"""

import os
import re
import argparse
from pathlib import Path

import torch
import anthropic
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Tiered thinking budget (tokens) keyed by Hendrycks MATH difficulty level.
# max_new_tokens per problem is set to 2x the thinking budget to leave enough
# room for the final answer after </think>.
# ---------------------------------------------------------------------------
THINKING_BUDGET = {
    1: 2048,
    2: 3072,
    3: 4096,
    4: 6144,
    5: 8192,
}


def sanitise_model_name(name: str) -> str:
    short = name.split("/")[-1]
    return re.sub(r"[^\w\-.]", "_", short)


# ---------------------------------------------------------------------------
# Generator system prompt
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLE = r"""Here is an example of how you should respond:

Question:
The president can be any one of the 20 members, and the vice-president can be any one of the 19 remaining members. How many ways can you pick a president and a vice president?

Answer:
Step 1: There are 20 choices for president.
Step 2: There are 19 choices for vice president.
Step 3: Multiply to count ordered pairs: 20 × 19 = 380.
Step 4: The answer is: \boxed{380}

Now solve the following problem in the same style:
"""



GENERATOR_SYSTEM_PROMPT = (
    "You are a meticulous math tutor. Solve each problem step by step, in contest style. "
    "Output each step on its own line numbered as Step 1:, Step 2:, etc. "
    "Finish with a final line: 'The answer is: \\boxed{<expression>}' (just the boxed expression). "
    "Do not include extra commentary or text outside this format.\n\n"
    + FEW_SHOT_EXAMPLE
)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_trace(directory: str, filename: str, content: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"PROBLEM:\n{content['problem']}\n\n")
        f.write(f"GENERATED REASONING:\n{content['reasoning']}\n\n")
        f.write(f"GENERATOR: {content['generator']}\n")
        f.write(f"DATASET: {content['dataset']}\n")
        f.write(f"PROBLEM_IDX: {content['problem_idx']}\n")
    print(f"  Saved -> {path}")


def load_trace(path: str) -> dict | None:
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
        "reasoning":   " ".join(reasoning_m.group(1).split()),
        "generator":   generator_m.group(1).strip() if generator_m else "unknown",
        "dataset":     dataset_m.group(1).strip()   if dataset_m   else "unknown",
        "problem_idx": int(idx_m.group(1))          if idx_m       else -1,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_problems(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    num_problems: int,
    sample_start: int,
) -> list[dict]:
    """
    Load `num_problems` problems starting deterministically at `sample_start`.
    Returns a list of dicts with {"problem": str, "solution": str, "level": int}.
    The "level" field is parsed from the Hendrycks MATH "Level N" string (defaults to 3).
    """
    print(f"\nLoading dataset: {dataset_name}" +
          (f" / {dataset_config}" if dataset_config else "") +
          f" [{dataset_split}], indices [{sample_start}, {sample_start + num_problems})")

    load_kwargs = dict(split=dataset_split)
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    end = min(sample_start + num_problems, len(ds))
    ds_subset = ds.select(range(sample_start, end))

    problems = []
    for ex in ds_subset:
        problem_text  = ex.get("problem") or ex.get("question") or ex.get("input") or ""
        solution_text = ex.get("solution") or ex.get("answer") or ex.get("output") or ""

        raw_level = ex.get("level", "Level 3")
        try:
            level_int = int(str(raw_level).replace("Level", "").strip())
        except ValueError:
            level_int = 3

        problems.append({"problem": problem_text, "solution": solution_text, "level": level_int})

    print(f"  Loaded {len(problems)} problems.")
    return problems


# ---------------------------------------------------------------------------
# Generator: Claude (sequential — uses API, not HF)
# ---------------------------------------------------------------------------

def build_claude_generator(api_key_file: str, model: str):
    os.environ["ANTHROPIC_API_KEY"] = open(api_key_file).read().strip()
    client = anthropic.Anthropic()

    def generate_batch(questions: list[str], levels: list[int]) -> list[str]:
        """Claude does not support true batching — calls are made sequentially."""
        results = []
        for question, level in zip(questions, levels):
            resp = client.messages.create(
                model=model,
                system=GENERATOR_SYSTEM_PROMPT,
                temperature=1.0,
                max_tokens=16000,
                messages=[{"role": "user", "content": question}],
            )
            blocks = getattr(resp, "content", []) or []
            parts = [getattr(b, "text", None) for b in blocks if getattr(b, "text", None)]
            results.append("\n".join(parts).strip())
        return results

    return generate_batch


# ---------------------------------------------------------------------------
# Generator: Qwen (HuggingFace, thinking mode, batched)
# ---------------------------------------------------------------------------

def build_qwen_generator(model_name: str):
    """
    Loads a Qwen Instruct model with thinking mode enabled.
    Returns a generate_batch function that processes multiple problems in parallel.

    Key details:
      - Left-padding is required for correct batched generation with decoder-only models.
      - <think> is injected by the chat template into the input prompt, so we prepend
        it back after slicing off the input tokens from the output.
      - max_new_tokens is set to 2x the largest thinking budget in the batch, giving
        ample room for the final answer after </think>.
    """
    print(f"Loading Qwen generator: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"   # required for batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    def generate_batch(questions: list[str], levels: list[int]) -> list[str]:
        # Build chat-formatted prompts, one per question.
        # Each prompt gets a thinking budget scaled to its difficulty level.
        texts = []
        max_new_tokens = 0
        for question, level in zip(questions, levels):
            budget = THINKING_BUDGET.get(level, 4096)
            max_new_tokens = max(max_new_tokens, budget * 2)
            messages = [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                thinking_budget=budget,
            )
            texts.append(text)

        budgets = [THINKING_BUDGET.get(l, 4096) for l in levels]
        print(f"  Batch size: {len(questions)} | "
              f"thinking budgets: {budgets} | "
              f"max_new_tokens: {max_new_tokens}")

        # Tokenize with left-padding so all sequences are aligned on the right
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off the input tokens and decode each sequence.
        # <think> is consumed by the chat template as part of the input prompt,
        # so we prepend it back to keep the saved trace self-contained.
        input_len = inputs.input_ids.shape[1]
        results = []
        for out in output_ids:
            new_ids = out[input_len:]
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append("<think>" + decoded)
        return results

    return generate_batch


def build_mistral_generator(model_name: str):
    """
    Loads a Mistral Instruct model for batched generation.
    Returns a generate_batch function compatible with the rest of the pipeline.

    Key differences from build_qwen_generator:
      - Mistral has no thinking mode: enable_thinking / thinking_budget are not passed
        to apply_chat_template, and no <think> prefix is prepended to outputs.
      - Mistral's chat template does not inject a generation prompt token by default,
        so add_generation_prompt=True is still required but no extra kwargs are needed.
      - max_new_tokens is set to 2x the largest thinking budget in the batch (reusing
        THINKING_BUDGET as a difficulty-scaled token budget, not a thinking budget).
    """
    print(f"Loading Mistral generator: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"   # required for batched decoder-only generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    def generate_batch(questions: list[str], levels: list[int]) -> list[str]:
        texts = []
        max_new_tokens = 0
        for question, level in zip(questions, levels):
            budget = THINKING_BUDGET.get(level, 4096)
            max_new_tokens = max(max_new_tokens, budget * 2)
            messages = [
                    {"role": "user", "content": GENERATOR_SYSTEM_PROMPT + "\n\n" + question},
                    #{"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    #{"role": "user",   "content": question},
            ]
            # Mistral-7B-Instruct-v0.3 chat template does not accept
            # enable_thinking or thinking_budget — omit them entirely.
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        budgets = [THINKING_BUDGET.get(l, 4096) for l in levels]
        print(f"  Batch size: {len(questions)} | "
              f"token budgets: {budgets} | "
              f"max_new_tokens: {max_new_tokens}")

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off input tokens and decode. Unlike Qwen, Mistral produces no
        # <think> block, so the decoded text is returned directly with no prefix.
        input_len = inputs.input_ids.shape[1]
        results = []
        for out in output_ids:
            new_ids = out[input_len:]
            decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append(decoded)
        return results

    return generate_batch

# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

def build_generators(args) -> dict[str, callable]:
    """
    Return {model_label: generate_batch_fn}.
    All generate_batch functions share the signature:
        generate_batch(questions: list[str], levels: list[int]) -> list[str]
    """
    generators = {}

    for gen_name in args.generators:
        if gen_name == "claude":
            if not args.api_key_file:
                raise ValueError("--api_key_file required for claude generator")
            label = sanitise_model_name(args.claude_model)
            generators[label] = build_claude_generator(args.api_key_file, args.claude_model)
            print(f"Generator 'claude' ({args.claude_model}) -> folder '{label}'")

        elif gen_name == "qwen":
            label = sanitise_model_name(args.qwen_model)
            generators[label] = build_qwen_generator(args.qwen_model)
            print(f"Generator 'qwen' ({args.qwen_model}) -> folder '{label}'")
        
        elif gen_name == "mistral":
            label = sanitise_model_name(args.mistral_model)
            generators[label] = build_mistral_generator(args.mistral_model)
            print(f"Generator 'mistral' ({args.mistral_model}) -> folder '{label}'")

        else:
            raise ValueError(f"Unknown generator: {gen_name}")

    return generators


# ---------------------------------------------------------------------------
# Formatting strategies
# ---------------------------------------------------------------------------

# --- Strategy 1: LLM formatter model ---

def build_model_formatter(model_name: str, max_new_tokens: int = 2048):
    print(f"\nLoading formatting model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    fmt_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    def format_reasoning(problem: str, raw_reasoning: str, template_prompt: str) -> str:
        user_content = (
            f"{template_prompt}\n\n"
            f"[Math Problem]:\n{problem}\n\n"
            f"[Solution]:\n{raw_reasoning}\n"
        )
        messages = [
            {"role": "system", "content": "You are a precise formatting assistant. Follow the template exactly."},
            {"role": "user",   "content": user_content},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = fmt_pipeline(text, return_full_text=False)
        return out[0]["generated_text"].strip()

    return format_reasoning


# --- Strategy 2: Regex step extraction ---

def extract_steps(text: str) -> list[str]:
    """
    Robustly extract steps from model output.
    Handles formats like:
      - "Step 1: ..."
      - "**Step 1:** ..."
      - "**Step 1: ...**"
      - "1. ..."  (fallback: numbered list)
    """
    step_pattern = re.compile(
        r'\*{0,2}Step\s*\d+\s*[:.]\*{0,2}(.*?)(?=\*{0,2}Step\s*\d+\s*[:.]\*{0,2}|$)',
        re.IGNORECASE | re.DOTALL
    )
    steps = [m.group(1).strip() for m in step_pattern.finditer(text)]

    if not steps:
        numbered_pattern = re.compile(
            r'^\d+\.\s+(.*?)(?=^\d+\.\s+|$)',
            re.MULTILINE | re.DOTALL
        )
        steps = [m.group(1).strip() for m in numbered_pattern.finditer(text)]

    steps = [re.sub(r'\s+', ' ', s).strip() for s in steps if s.strip()]
    return steps


def extract_steps_qwen3(text: str) -> list[str]:
    # Normalise "--- ###" (collapsed) and "---\n###" (original) into "\n###"
    text = re.sub(r'-{3,}\s*#{1,3}', '\n###', text)
    text = re.sub(r'(?<!\n)#{1,3}', '\n###', text)

    steps = []

    # --- Pattern A: individual ### Step N: or ### **Method N:** headers ---
    step_pattern = re.compile(
        r'#{1,3}\s*\*{0,2}(?:Step|Method)\s*\d+\s*[:.]\*{0,2}(.*?)(?=#{1,3}\s*\*{0,2}(?:Step|Method)\s*\d+\s*[:.] |#{1,3}.*?(?:Final Answer|Conclusion)|$)',
        re.IGNORECASE | re.DOTALL
    )
    steps = [m.group(1).strip() for m in step_pattern.finditer(text)]

    # --- Pattern B: ### Step-by-step: followed by a numbered list ---
    if not steps:
        stepbystep_pattern = re.compile(
            r'#{1,3}\s*\*{0,2}Step.by.step[^:\n]*[:.]\*{0,2}(.*?)(?=#{1,3}.*?(?:Final Answer|Conclusion)|$)',
            re.IGNORECASE | re.DOTALL
        )
        m = stepbystep_pattern.search(text)
        if m:
            block = m.group(1).strip()
            numbered_pattern = re.compile(
                r'^\d+\.\s+(.*?)(?=^\d+\.\s+|$)',
                re.MULTILINE | re.DOTALL
            )
            steps = [m.group(1).strip() for m in numbered_pattern.finditer(block)]

    # --- Pattern C: any ### descriptive header sections ---
    if not steps:
        section_pattern = re.compile(
            r'#{1,3}\s*\*{0,2}(?!.*?(?:Final Answer|Conclusion))([^#\n]+)\*{0,2}\n(.*?)(?=#{1,3}|$)',
            re.IGNORECASE | re.DOTALL
        )
        for m in section_pattern.finditer(text):
            header = m.group(1).strip().strip('*').strip().rstrip(':')
            body = m.group(2).strip()
            if body:
                steps.append(f"{header}\n{body}")

    # --- Fallback: plain numbered list anywhere in text ---
    if not steps:
        numbered_pattern = re.compile(
            r'^\d+\.\s+(.*?)(?=^\d+\.\s+|$)',
            re.MULTILINE | re.DOTALL
        )
        steps = [m.group(1).strip() for m in numbered_pattern.finditer(text)]

    # --- Always append Final Answer / Conclusion as last step if present ---
    # Use \s* instead of \n to handle both collapsed and multi-line traces
    final_pattern = re.compile(
        r'#{1,3}\s*[\*✅]*\s*(?:Final Answer|Final Conclusion|Conclusion)\*{0,2}\s*:?\s*(.*?)$',
        re.IGNORECASE | re.DOTALL
    )
    final_match = final_pattern.search(text)
    if final_match:
        final_text = final_match.group(1).strip()
        if final_text:
            steps.append(final_text)

    return [re.sub(r'\s+', ' ', s).strip() for s in steps if s.strip()]


def build_steps_formatter(args):
    model_name = sanitise_model_name(args.qwen_model if "qwen" in args.generators else "").lower()
    if model_name.startswith("qwen3"):
        print("\n[Formatter] Using Qwen3-specific step extraction.")
        step_extractor = extract_steps_qwen3
    else:
        print("\n[Formatter] Using generic regex step extraction.")
        step_extractor = extract_steps

    def format_reasoning(problem: str, raw_reasoning: str, template_prompt: str) -> str:
        # Only extract from the final answer, not the thinking trace
        if "<think>" in raw_reasoning and "</think>" in raw_reasoning:
            text = raw_reasoning.split("</think>")[-1].strip()
        else:
            text = raw_reasoning

        steps = step_extractor(text)
        if not steps:
            print("  Warning: no steps found via regex; returning post-think text.")
            return text
        formatted = "\n\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        print(f"  Extracted {len(steps)} steps via regex.")
        return formatted

    return format_reasoning


# --- Strategy 3: <think> token + newline splitting ---

def build_thinking_formatter():
    def format_reasoning(problem: str, raw_reasoning: str, template_prompt: str) -> str:
        if "<think>" not in raw_reasoning:
            print("  Warning: no <think> token found; returning raw reasoning.")
            return raw_reasoning

        reasoning_lines = raw_reasoning.split("<think>")[1].split('\n\n')
        reasoning_lines[0] = reasoning_lines[0].lstrip('\n')
        segments = [seg.strip() for seg in reasoning_lines if seg.strip()]
        formatted = "\n\n".join(segments)
        print(f"  Split into {len(segments)} paragraph(s) via <think> + newlines.")
        return formatted

    return format_reasoning


# ---------------------------------------------------------------------------
# Formatter factory
# ---------------------------------------------------------------------------

def build_formatter(args):
    """
    Instantiate and return the appropriate format_reasoning callable based
    on --format_mode.  All returned callables share the same signature:
        format_reasoning(problem: str, raw_reasoning: str, template_prompt: str) -> str
    """
    mode = args.format_mode

    if mode == "model":
        return build_model_formatter(args.formatter_model, args.formatter_max_new_tokens)
    elif mode == "steps":
        print("\n[Formatter] Using regex step-extraction (no model needed).")
        return build_steps_formatter(args)
    elif mode == "thinking":
        print("\n[Formatter] Using <think>-token + newline splitting (no model needed).")
        return build_thinking_formatter()
    else:
        raise ValueError(f"Unknown --format_mode: {mode!r}. "
                         "Choose from: model, steps, thinking")


# ---------------------------------------------------------------------------
# Generation pass (batched)
# ---------------------------------------------------------------------------

def run_generation(
    generators: dict[str, callable],
    problems: list[dict],
    save_reasoning: str,
    dataset_tag: str,
    sample_start: int,
    batch_size: int,
) -> None:
    """
    Run all generators on all problems in batches and save raw reasoning traces.
    Traces are stored at:
        {save_reasoning}/{model_label}/{dataset_tag}_{global_idx}.txt

    Already-completed traces are skipped. Remaining problems are grouped into
    batches of `batch_size` and processed in parallel on the GPU.
    """
    for model_label, generate_batch_fn in generators.items():
        gen_dir = os.path.join(save_reasoning, model_label)
        print(f"\n{'='*60}")
        print(f"Generator: {model_label}  ->  {gen_dir}")
        print(f"{'='*60}")

        # Collect problems that still need to be generated
        pending = []
        for local_i, prob in enumerate(problems):
            global_idx = sample_start + local_i
            out_path = os.path.join(gen_dir, f"{dataset_tag}_{global_idx}.txt")
            if os.path.exists(out_path):
                print(f"  Already exists, skipping: {dataset_tag}_{global_idx}.txt")
            else:
                pending.append((local_i, global_idx, prob))

        if not pending:
            print("  All traces already exist, nothing to generate.")
            continue

        # Process in batches
        num_batches = (len(pending) + batch_size - 1) // batch_size
        for batch_num, batch_start in enumerate(range(0, len(pending), batch_size)):
            batch   = pending[batch_start : batch_start + batch_size]
            questions = [p["problem"]           for _, _, p in batch]
            levels    = [int(p.get("level", 3)) for _, _, p in batch]
            indices   = [g                      for _, g, _ in batch]

            print(f"\n  Batch {batch_num + 1}/{num_batches} — "
                  f"problem indices {indices}, levels {levels}")

            raw_outputs = generate_batch_fn(questions, levels)

            for (local_i, global_idx, prob), raw in zip(batch, raw_outputs):
                print(f"  Generated {len(raw.split())} words for idx {global_idx}.")
                save_trace(gen_dir, f"{dataset_tag}_{global_idx}.txt", {
                    "problem":     prob["problem"],
                    "reasoning":   raw,
                    "generator":   model_label,
                    "dataset":     dataset_tag,
                    "problem_idx": global_idx,
                })


# ---------------------------------------------------------------------------
# Formatting pass
# ---------------------------------------------------------------------------

def run_formatting(
    format_fn: callable,
    template_prompt: str,
    save_reasoning: str,
    save_formatted: str,
    model_labels: list[str],
    dataset_tag: str,
    sample_start: int,
    num_problems: int,
) -> None:
    """
    For each generator's raw traces, call the formatting function and save results.
    Formatted traces are stored at:
        {save_formatted}/{model_label}/{dataset_tag}_{global_idx}.txt
    """
    for model_label in model_labels:
        raw_dir = os.path.join(save_reasoning, model_label)
        fmt_dir = os.path.join(save_formatted, model_label)
        os.makedirs(fmt_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Formatting: {model_label}  ->  {fmt_dir}")
        print(f"{'='*60}")

        for i in range(num_problems):
            global_idx = sample_start + i
            src_file = os.path.join(raw_dir, f"{dataset_tag}_{global_idx}.txt")
            dst_file = os.path.join(fmt_dir, f"{dataset_tag}_{global_idx}.txt")

            if not os.path.exists(src_file):
                print(f"  Raw trace not found, skipping: {src_file}")
                continue

            if os.path.exists(dst_file):
                print(f"  Already formatted, skipping: {dst_file}")
                continue

            trace = load_trace(src_file)
            if trace is None:
                continue

            print(f"\n  [{i+1}/{num_problems}] Formatting idx {global_idx} ...")
            formatted = format_fn(trace["problem"], trace["reasoning"], template_prompt)
            print(f"  Formatted to {len(formatted.split())} words.")

            save_trace(fmt_dir, f"{dataset_tag}_{global_idx}.txt", {
                "problem":     trace["problem"],
                "reasoning":   formatted,
                "generator":   trace["generator"],
                "dataset":     trace["dataset"],
                "problem_idx": trace["problem_idx"],
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-generator math reasoning collector + formatter."
    )

    # --- Generator selection ---
    ap.add_argument(
        "--generators", nargs="+",
        choices=["claude", "qwen", "mistral"],
        default=["qwen"],
        help="One or more generator models to run.",
    )
    ap.add_argument("--api_key_file",
                    help="Path to Anthropic API key file (required for claude generator).")
    ap.add_argument("--claude_model", default="claude-3-7-sonnet-20250219",
                    help="Claude model string.")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-Math-7B-Instruct",
                    help="HuggingFace model name for qwen generator.")
    ap.add_argument("--mistral_model", default="mistralai/Mistral-7B-Instruct-v0.3",
                    help="HuggingFace model name for mistral generator.")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Number of problems to generate in parallel (Qwen only). "
                         "Reduce if you run out of GPU memory.")

    # --- Dataset ---
    ap.add_argument("--dataset_name", default="EleutherAI/hendrycks_math",
                    help="HuggingFace dataset identifier.")
    ap.add_argument("--dataset_config", default="algebra",
                    help="Dataset config/subset name (e.g. 'algebra', 'geometry').")
    ap.add_argument("--dataset_split", default="test",
                    help="Dataset split (default: test).")
    ap.add_argument("--num_problems", type=int, default=5,
                    help="Number of problems to use.")
    ap.add_argument("--sample_start", type=int, default=0,
                    help="Start index into the dataset (deterministic offset).")

    # --- Formatting mode ---
    ap.add_argument(
        "--format_mode",
        choices=["model", "steps", "thinking"],
        default="model",
        help=(
            "Formatting strategy to apply to raw reasoning traces:\n"
            "  model    – reformat using a HuggingFace LLM (default; requires --formatter_model)\n"
            "  steps    – regex extraction of numbered/labelled reasoning steps\n"
            "  thinking – split on <think> token then on double-newlines"
        ),
    )

    # --- Formatter model options (only used when --format_mode model) ---
    ap.add_argument("--template_file", default="reformat.txt",
                    help="Path to the formatting template/instruction file (model mode only).")
    ap.add_argument("--formatter_model", default="Qwen/Qwen2.5-72B-Instruct",
                    help="HuggingFace model used to reformat reasoning traces (model mode only).")
    ap.add_argument("--formatter_max_new_tokens", type=int, default=2048,
                    help="Max new tokens for the formatting model (model mode only).")

    # --- Paths ---
    ap.add_argument("--save_reasoning", default="./out/raw",
                    help="Root directory for raw reasoning traces.")
    ap.add_argument("--save_formatted", default="./out/formatted",
                    help="Root directory for reformatted reasoning traces.")

    # --- Control flow ---
    ap.add_argument("--skip_generation", action="store_true",
                    help="Skip generation; load existing raw traces and only format them.")
    ap.add_argument("--skip_formatting", action="store_true",
                    help="Skip the formatting step (only generate and save raw traces).")

    args = ap.parse_args()

    dataset_tag = (args.dataset_config or args.dataset_name.split("/")[-1]).replace("/", "_")

    model_labels = []
    for gen_name in args.generators:
        if gen_name == "claude":
            model_labels.append(sanitise_model_name(args.claude_model))
        elif gen_name == "qwen":
            model_labels.append(sanitise_model_name(args.qwen_model))
        elif gen_name == "mistral":
            model_labels.append(sanitise_model_name(args.mistral_model))

    # ------------------------------------------------------------------
    # 1. Generation
    # ------------------------------------------------------------------
    if not args.skip_generation:
        problems = load_problems(
            args.dataset_name,
            args.dataset_config,
            args.dataset_split,
            args.num_problems,
            args.sample_start,
        )
        generators = build_generators(args)
        run_generation(
            generators,
            problems,
            args.save_reasoning,
            dataset_tag,
            args.sample_start,
            args.batch_size,
        )
    else:
        print("\n[Generation skipped -- loading from existing raw traces]")

    # ------------------------------------------------------------------
    # 2. Formatting
    # ------------------------------------------------------------------
    if not args.skip_formatting:
        if args.format_mode == "model":
            if not os.path.exists(args.template_file):
                raise FileNotFoundError(
                    f"Template file not found: {args.template_file}\n"
                    "Create a reformat.txt with formatting instructions, "
                    "or switch to --format_mode steps or --format_mode thinking."
                )
            template_prompt = Path(args.template_file).read_text(encoding="utf-8").strip()
            print(f"\nTemplate loaded from {args.template_file} ({len(template_prompt)} chars).")
        else:
            template_prompt = ""

        format_fn = build_formatter(args)
        run_formatting(
            format_fn,
            template_prompt,
            args.save_reasoning,
            args.save_formatted,
            model_labels,
            dataset_tag,
            args.sample_start,
            args.num_problems,
        )
    else:
        print("\n[Formatting skipped]")

    print("\nDone.")


if __name__ == "__main__":
    main()