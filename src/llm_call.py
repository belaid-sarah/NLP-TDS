"""
llm_call.py – TD3 : extraction de noms de comiques via LLM
===========================================================
Deux approches :
  1. Structured output  (video_name_to_comic_names)   : str  → list[str]
  2. CSV prompt         (video_names_to_comic_names)  : list[str] → list[list[str]]

Plusieurs variantes de prompts sont disponibles dans PROMPT_VARIANTS.
Lancer avec --compare pour afficher un diff côte à côte sur un échantillon.

Usage :
    python llm_call.py                        # variante "few_shot" par défaut
    python llm_call.py --variant zero_shot
    python llm_call.py --compare              # compare toutes les variantes sur 10 titres
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import openai
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration API
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    raise RuntimeError("Missing API key. Set the GROQ_API_KEY environment variable.")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# ---------------------------------------------------------------------------
# Models & pricing  (USD / 1 M tokens)
# ---------------------------------------------------------------------------
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama3-8b-8192":          (0.05, 0.08),
    "gemma2-9b-it":            (0.20, 0.20),
}

SINGLE_MODEL = "llama-3.3-70b-versatile"  # structured output
BATCH_MODEL  = "llama-3.3-70b-versatile"  # CSV prompt

# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------
#  Each entry:  { "description": str, "system": str | None, "template": Callable[[list[str]], str] }

_CSV_HEADER_INSTRUCTION = (
    "Start your reply with ```csv\n"
    "video_name;comic_names\n"
    "One row per video in the same order. "
    "Separate multiple names with |. "
    "Leave comic_names empty if no comedian found. "
    "Do NOT add any text outside the CSV block."
)

_FEW_SHOT_EXAMPLES = (
    "Examples (→ means 'answer'):\n"
    "  Gad Elmaleh - L'autre c'est moi (spectacle complet) → Gad Elmaleh\n"
    "  Jamel Debbouze - Sans Filet Live → Jamel Debbouze\n"
    "  Kaamelott - Livre I (compilation) → (empty)\n"
    "  Florence Foresti et Thomas VDB en direct → Florence Foresti|Thomas VDB\n"
)

_SYSTEM_EXPERT = (
    "You are an expert in French and international stand-up comedy. "
    "You have encyclopedic knowledge of comedian names and excel at "
    "extracting structured information from text."
)


def _tmpl_zero_shot(titles: list[str]) -> str:
    """No examples, no chain-of-thought – minimal instruction."""
    return (
        "For each video title below, extract all comedian / humorist names.\n"
        + _CSV_HEADER_INSTRUCTION
        + "\n\n"
        + "\n".join(titles)
    )


def _tmpl_chain_of_thought(titles: list[str]) -> str:
    """Ask the model to reason step-by-step before producing the CSV."""
    return (
        "For each video title below, extract all comedian / humorist names.\n"
        "Think step by step before answering (Chain-of-Thought).\n"
        + _CSV_HEADER_INSTRUCTION
        + "\n\n"
        + "\n".join(titles)
    )


def _tmpl_few_shot(titles: list[str]) -> str:
    """Four in-context examples before the real titles."""
    return (
        "For each video title below, extract all comedian / humorist names.\n\n"
        + _FEW_SHOT_EXAMPLES
        + "\n"
        + _CSV_HEADER_INSTRUCTION
        + "\n\n"
        + "\n".join(titles)
    )


def _tmpl_structured_instruct(titles: list[str]) -> str:
    """Explicit NER task framing with rules."""
    return (
        "Task: Named Entity Recognition – extract comedian names from YouTube video titles.\n"
        "Rules:\n"
        "  - Only include real stand-up comedians / humorists.\n"
        "  - Ignore directors, TV show names, or band names.\n"
        "  - If uncertain, leave the field empty.\n\n"
        + _CSV_HEADER_INSTRUCTION
        + "\n\n"
        + "\n".join(titles)
    )


def _tmpl_role_play(titles: list[str]) -> str:
    """Ask the model to adopt an expert persona inline (no system prompt)."""
    return (
        "Act as an expert in French stand-up comedy with encyclopedic knowledge "
        "of all comedians who have ever performed.\n"
        "For each video title below, extract all comedian / humorist names.\n"
        + _CSV_HEADER_INSTRUCTION
        + "\n\n"
        + "\n".join(titles)
    )


PROMPT_VARIANTS: dict[str, dict] = {
    "zero_shot": {
        "description": "No examples, no CoT, no system prompt – baseline.",
        "system": None,
        "template": _tmpl_zero_shot,
    },
    "chain_of_thought": {
        "description": "Ask the model to reason step-by-step before the CSV.",
        "system": None,
        "template": _tmpl_chain_of_thought,
    },
    "few_shot": {
        "description": "Four in-context examples before the titles.",
        "system": None,
        "template": _tmpl_few_shot,
    },
    "structured_instruct": {
        "description": "Explicit NER task framing with rules.",
        "system": None,
        "template": _tmpl_structured_instruct,
    },
    "system_expert": {
        "description": "Expert system prompt + zero-shot user message.",
        "system": _SYSTEM_EXPERT,
        "template": _tmpl_zero_shot,
    },
    "system_expert_few_shot": {
        "description": "Expert system prompt + four in-context examples.",
        "system": _SYSTEM_EXPERT,
        "template": _tmpl_few_shot,
    },
    "role_play": {
        "description": "Inline role-play persona (no system prompt).",
        "system": None,
        "template": _tmpl_role_play,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _usage(reply) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from a completion reply."""
    u = getattr(reply, "usage", None)
    if not u:
        return 0, 0
    inp = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0)
    out = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0)
    return inp or 0, out or 0


def _compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICING_PER_1M.get(model)
    if not prices:
        return 0.0
    ip, op = prices
    return (input_tokens / 1_000_000) * ip + (output_tokens / 1_000_000) * op


def _make_metrics(model: str, inp: int, out: int, latency: float) -> dict:
    return {
        "input_tokens":  inp,
        "output_tokens": out,
        "total_tokens":  inp + out,
        "cost_usd":      round(_compute_cost_usd(model, inp, out), 8),
        "latency_s":     round(latency, 3),
    }


def _extract_csv_block(text: str) -> str:
    """Pull the content between ```csv … ``` fences."""
    lines = text.splitlines()
    in_csv, csv_lines = False, []
    for line in lines:
        if line.strip().lower().startswith("```csv"):
            in_csv = True
            continue
        if in_csv and line.strip().startswith("```"):
            break
        if in_csv:
            csv_lines.append(line)
    return "\n".join(csv_lines) if csv_lines else text


def _parse_csv_names(text: str, expected_rows: int) -> list[list[str]]:
    """Parse the CSV block and return one list[str] of names per video."""
    csv_text = _extract_csv_block(text)
    lines = [l.strip() for l in csv_text.splitlines() if l.strip()]

    # Skip the header row
    start = 0
    for i, line in enumerate(lines):
        if line.replace(" ", "").lower() == "video_name;comic_names":
            start = i + 1
            break

    results: list[list[str]] = []
    for line in lines[start : start + expected_rows]:
        if ";" not in line:
            results.append([])
            continue
        right = line.split(";", 1)[1].strip()
        results.append([n.strip() for n in right.split("|") if n.strip()])

    while len(results) < expected_rows:
        results.append([])
    return results


def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ---------------------------------------------------------------------------
# 1.  Structured output – single video title
# ---------------------------------------------------------------------------

class ComicNames(BaseModel):
    """Pydantic schema for structured output."""
    names: list[str] = Field(
        default_factory=list,
        description=(
            "List of comedian / humorist names found in the video title. "
            "Empty list if none are found."
        ),
    )


def video_name_to_comic_names(video_name: str) -> tuple[list[str], dict]:
    """
    Structured output: one video title → list of comedian names + metrics.

    Uses ``response_format=ComicNames`` so the model is constrained to return
    a valid JSON object matching the Pydantic schema.

    Parameters
    ----------
    video_name : str
        The YouTube video title.

    Returns
    -------
    names   : list[str]   – comedian names (empty list if none found)
    metrics : dict        – input_tokens, output_tokens, cost_usd, latency_s
    """
    t0 = time.perf_counter()
    reply = client.chat.completions.parse(
        model=SINGLE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at identifying comedian / humorist names "
                    "in YouTube video titles. "
                    "Return an empty list if no comedian name is found."
                ),
            },
            {
                "role": "user",
                "content": f"Extract all comedian names from this video title: '{video_name}'",
            },
        ],
        response_format=ComicNames,
        temperature=0,
    )
    latency = time.perf_counter() - t0
    parsed = reply.choices[0].message.parsed
    names = parsed.names if parsed else []
    inp, out = _usage(reply)
    return names, _make_metrics(SINGLE_MODEL, inp, out, latency)


# ---------------------------------------------------------------------------
# 2.  CSV prompt – batch of video titles
# ---------------------------------------------------------------------------

def video_names_to_comic_names(
    video_names: list[str],
    variant: str = "zero_shot",
) -> tuple[list[list[str]], dict]:
    """
    CSV-prompt approach: list of video titles → list of comedian-name lists + metrics.

    The model is asked to reply starting with a CSV block:
        ```csv
        video_name;comic_names
        …
    The ``variant`` parameter selects which prompt strategy to use
    (see :data:`PROMPT_VARIANTS`).

    Parameters
    ----------
    video_names : list[str]
        Video titles to process.
    variant : str
        Key in :data:`PROMPT_VARIANTS`. Defaults to ``"zero_shot"``.

    Returns
    -------
    names_per_video : list[list[str]]
    metrics         : dict  – input_tokens, output_tokens, cost_usd, latency_s
    """
    if not video_names:
        return [], _make_metrics(BATCH_MODEL, 0, 0, 0.0)

    cfg = PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["zero_shot"])
    messages: list[dict] = []
    if cfg["system"]:
        messages.append({"role": "system", "content": cfg["system"]})
    messages.append({"role": "user", "content": cfg["template"](video_names)})

    t0 = time.perf_counter()
    reply = client.chat.completions.create(
        model=BATCH_MODEL,
        messages=messages,
        temperature=0,
    )
    latency = time.perf_counter() - t0

    raw_text = reply.choices[0].message.content or ""
    names = _parse_csv_names(raw_text, len(video_names))
    inp, out = _usage(reply)
    return names, _make_metrics(BATCH_MODEL, inp, out, latency)


# ---------------------------------------------------------------------------
# 3.  Compare all variants on a shared sample
# ---------------------------------------------------------------------------

def compare_variants(
    video_names: list[str],
    variants: list[str] | None = None,
) -> dict[str, dict]:
    """
    Run several prompt variants on the same list of video titles.

    Parameters
    ----------
    video_names : list[str]
        Titles to use as the test set.
    variants : list[str] | None
        Subset of :data:`PROMPT_VARIANTS` keys. All variants if *None*.

    Returns
    -------
    results : dict  variant_name → { "description", "names", "metrics" }
    """
    if variants is None:
        variants = list(PROMPT_VARIANTS.keys())

    results: dict[str, dict] = {}
    for v in variants:
        names, metrics = video_names_to_comic_names(video_names, variant=v)
        results[v] = {
            "description": PROMPT_VARIANTS[v]["description"],
            "names":        names,
            "metrics":      metrics,
        }
        print(
            f"[{v:26s}]  tokens={metrics['total_tokens']:5d}"
            f"  cost=${metrics['cost_usd']:.6f}"
            f"  latency={metrics['latency_s']:.2f}s"
        )
    return results


# ---------------------------------------------------------------------------
# Helpers for main
# ---------------------------------------------------------------------------

def _resolve_input_csv_path(project_root: Path) -> Path:
    default = project_root / "names_train.csv"
    if default.exists():
        return default
    candidates = sorted(project_root.glob("names_train*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No input CSV found in {project_root}. Expected names_train.csv or names_train*.csv."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TD3 – LLM comic-name extractor")
    parser.add_argument(
        "--variant",
        default="few_shot",
        choices=list(PROMPT_VARIANTS.keys()),
        help="Prompt variant to use for batch prediction (default: few_shot).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare ALL variants on the first 10 comic titles, then exit.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=40,
        help="Number of titles per API call (default: 40).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    in_path = _resolve_input_csv_path(project_root)

    rows: list[dict[str, str]] = []
    with in_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=","):
            rows.append(row)

    titles   = [row["video_name"] for row in rows]
    is_comic = [int(row.get("is_comic", 0) or 0) for row in rows]
    idxs     = [i for i, y in enumerate(is_comic) if y == 1]
    titles_to_predict = [titles[i] for i in idxs]

    # ------------------------------------------------------------------
    # Comparison mode: diff all variants on a small sample, then exit
    # ------------------------------------------------------------------
    if args.compare:
        sample = titles_to_predict[:10]
        print(f"\n=== Comparing {len(PROMPT_VARIANTS)} variants on {len(sample)} titles ===\n")
        results = compare_variants(sample)

        print("\n" + "=" * 80)
        print("Per-title diff")
        print("=" * 80)
        for i, title in enumerate(sample):
            print(f"\n  [{i:02d}] {title}")
            for v, r in results.items():
                names_str = ", ".join(r["names"][i]) if r["names"][i] else "(none)"
                print(f"         {v:26s}: {names_str}")

        print("\n" + "=" * 80)
        print("Metrics summary")
        print("=" * 80)
        print(f"  {'variant':<26}  {'in_tok':>6}  {'out_tok':>7}  {'cost_usd':>10}  {'latency_s':>9}")
        print(f"  {'-'*26}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*9}")
        for v, r in results.items():
            m = r["metrics"]
            print(
                f"  {v:<26}  {m['input_tokens']:>6}  {m['output_tokens']:>7}"
                f"  {m['cost_usd']:>10.6f}  {m['latency_s']:>9.2f}"
            )
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Full prediction with selected variant
    # ------------------------------------------------------------------
    out_path = project_root / "td3_predictions.csv"
    preds: list[list[str]] = [[] for _ in titles]
    total_in, total_out, total_time = 0, 0, 0.0

    for batch_num, batch in enumerate(_chunks(titles_to_predict, args.batch_size), start=1):
        batch_preds, metrics = video_names_to_comic_names(batch, variant=args.variant)
        total_time += metrics["latency_s"]
        total_in   += metrics["input_tokens"]
        total_out  += metrics["output_tokens"]
        print(
            f"  batch {batch_num:03d}  titles={len(batch):3d}"
            f"  tokens={metrics['total_tokens']:5d}"
            f"  cost=${metrics['cost_usd']:.6f}"
        )
        for j, names in enumerate(batch_preds):
            original_idx = idxs[(batch_num - 1) * args.batch_size + j]
            preds[original_idx] = names

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["video_name", "comic_names"])
        for title, names in zip(titles, preds):
            w.writerow([title, "|".join(names)])

    print("\nDONE")
    print(f"  variant        : {args.variant}")
    print(f"  input_file     : {in_path}")
    print(f"  output_file    : {out_path}")
    print(f"  input_tokens   : {total_in}")
    print(f"  output_tokens  : {total_out}")
    print(f"  cost_usd       : {_compute_cost_usd(BATCH_MODEL, total_in, total_out):.6f}")
    print(f"  latency_s      : {total_time:.2f}")
