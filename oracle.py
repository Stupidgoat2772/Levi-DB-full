#!/usr/bin/env python3
"""oracle.py — Bible Study Oracle.

Multi-pass semantic retrieval from Levi DB + local LLM synthesis.
Answers questions grounded in biblical text with cross-reference discovery.

Retrieval Strategy (3 passes):
  Pass 1 — Direct semantic search on your question (primary results)
  Pass 2 — Strong's-based expansion: extract lexical roots from Pass 1,
            re-query to find verses sharing the same Hebrew/Greek roots
  Pass 3 — Cross-testament sweep: if Pass 1 is mostly OT, query NT for
            the same theme, and vice versa. Surfaces typological links.

All results are ranked by (distance × weight) and deduplicated before
being sent to the LLM.

Usage:
    oracle.py "What does it mean that God rested on the seventh day?"
    oracle.py "What is the heart?" --model qwen2.5:14b
    oracle.py "Explain atonement" --n 10 --testament OT
    oracle.py "Who is the Word?" --no-cross   # skip cross-ref passes
    oracle.py "Fear of the Lord" --raw         # print context only, no LLM
"""

import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(os.environ.get("LEVI_DATA_DIR", Path(__file__).resolve().parent / "data"))
DB_PATH = ROOT / "chromadb" / "levi.db"

# ---------------------------------------------------------------------------
# Retrieval (SQLite + numpy brute-force cosine similarity)
# ---------------------------------------------------------------------------

def _ensure_deps():
    try:
        import numpy as np
        import sentence_transformers
        return np, sentence_transformers
    except ImportError:
        print("ERROR: missing dependency.\n  pip install numpy sentence-transformers", file=sys.stderr)
        sys.exit(1)


def query_db(text: str, n: int, testament: str = None) -> list[dict]:
    """Semantic search against Levi DB. Returns list of hit dicts."""
    # Import here so oracle.py can be imported without triggering deps check
    from ask_levi import query
    return query(text, n=n, testament=testament)


# Domains that are noise for cross-reference queries — proper names, genealogies
_NOISE_DOMAINS = {
    "Names of Locations", "Names of People", "Names of Deities",
    "Names of Groups", "Names of Constructions",
}

def extract_strongs(text: str) -> list[str]:
    """Pull semantically significant Strong's IDs from enriched chunk text.

    Skips roots that only appear in name/location domains — those produce
    noise in the cross-reference pass (genealogy lists, territory surveys).
    Keeps roots that carry theological/semantic weight.
    """
    # Find all Strong's IDs
    all_ids = re.findall(r'[HG]\d{1,5}', text)

    # Find sections tagged as name-only domains and their Strong's IDs
    # Format in text: [Domain: Names of Locations] or similar
    noise_ids = set()
    # Walk word blocks: "word [Hnnnn] (def) [Domain: X]"
    # A root is noise if ALL its domain tags are noise domains
    word_blocks = re.split(r'\|\s*', text)
    for block in word_blocks:
        strongs_in_block = re.findall(r'[HG]\d{1,5}', block)
        domain_match = re.search(r'\[Domain:\s*([^\]]+)\]', block)
        if domain_match and strongs_in_block:
            domains = {d.strip() for d in domain_match.group(1).split(',')}
            if domains and domains.issubset(_NOISE_DOMAINS):
                noise_ids.update(strongs_in_block)

    return [sid for sid in all_ids if sid not in noise_ids]


def dominant_testament(hits: list[dict]) -> str:
    """Return the testament that appears most in a hit list."""
    counts = {"OT": 0, "NT": 0}
    for h in hits:
        t = h.get("testament", "")
        if t in counts:
            counts[t] += 1
    return "NT" if counts["OT"] >= counts["NT"] else "OT"


def multi_pass_retrieve(question: str, n_primary: int = 7, n_cross: int = 5,
                        testament: str = None, do_cross: bool = True) -> list[dict]:
    """
    Three-pass retrieval. Returns merged, deduplicated, weighted hit list.

    Pass 1: Direct semantic search.
    Pass 2: Strong's-based re-query (finds lexically related verses).
    Pass 3: Cross-testament sweep (OT↔NT typological links).
    """
    if not DB_PATH.exists():
        print("ERROR: Levi DB not found. Run build_levi_db.py first.", file=sys.stderr)
        sys.exit(1)

    seen = {}  # id → hit (keep lowest distance)

    # --- Pass 1: Direct ---
    p1 = query_db(question, n_primary, testament)
    for h in p1:
        h["weight"] = 1.0
        h["passes"] = ["primary"]
        seen[h["id"]] = h

    if not do_cross:
        return sorted(seen.values(), key=lambda x: x["distance"] * x["weight"])

    # --- Pass 2: Strong's expansion ---
    all_strongs = []
    for h in p1:
        all_strongs.extend(extract_strongs(h["text"]))

    # Deduplicate and take top 5 most common roots
    from collections import Counter
    top_roots = [root for root, _ in Counter(all_strongs).most_common(5)]

    for root in top_roots:
        p2_hits = query_db(root, n_cross, testament)
        for h in p2_hits:
            if h["id"] not in seen:
                h["weight"] = 1.1  # slight penalty — secondary source
                h["passes"] = ["strongs"]
                seen[h["id"]] = h
            else:
                # Found in both passes — boost it (convergent signal)
                seen[h["id"]]["weight"] = max(0.85, seen[h["id"]]["weight"] - 0.15)
                seen[h["id"]]["passes"].append("strongs")

    # --- Pass 3: Cross-testament sweep ---
    if testament is None:
        flip = {"OT": "NT", "NT": "OT"}
        primary_testament = dominant_testament(p1)
        cross_testament = flip.get(primary_testament)

        if cross_testament:
            p3_hits = query_db(question, n_cross, cross_testament)
            for h in p3_hits:
                if h["id"] not in seen:
                    h["weight"] = 1.2  # lower priority than primary
                    h["passes"] = ["cross-testament"]
                    seen[h["id"]] = h
                else:
                    seen[h["id"]]["weight"] = max(0.8, seen[h["id"]]["weight"] - 0.2)
                    seen[h["id"]]["passes"].append("cross-testament")

    # Final rank: distance × weight (lower = better)
    ranked = sorted(seen.values(), key=lambda x: x["distance"] * x["weight"])
    return ranked


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Bible Study Oracle — a rigorous theological reasoning engine.

You answer questions about biblical truth grounded solely in the scripture passages provided to you. You have access to enriched Hebrew and Greek lexical data alongside each verse.

Rules:
- Ground every claim in the provided passages. If the passages don't support a claim, say so.
- Cite verse references explicitly (e.g., Gen.1.1, John.1.1).
- When the same concept appears in both Old and New Testament passages, name it as a cross-reference and explain the connection.
- Distinguish between what the text directly states and what is theological inference.
- Use the Strong's definitions and semantic domain data in the passages to explain word meaning where relevant.
- If passages create tension or apparent contradiction, name the tension honestly and reason through it.
- No devotional fluff. No filler. Precise, grounded, dense.
- Structure your answer: Primary Answer → Supporting Passages → Cross-References (if any) → Key Word Analysis (if relevant)."""


def format_context(hits: list[dict], max_hits: int = 12) -> str:
    """Format retrieved hits as LLM context block."""
    lines = ["=== RETRIEVED SCRIPTURE PASSAGES ===\n"]
    for i, h in enumerate(hits[:max_hits], 1):
        passes = ", ".join(h.get("passes", []))
        lines.append(f"[{i}] {h['id']} ({h['testament']}) | dist: {h['distance']:.4f} | source: {passes}")
        lines.append(h["text"])
        lines.append("")
    return "\n".join(lines)


def print_raw(hits: list[dict], max_hits: int = 12):
    """Print raw retrieval results without LLM synthesis."""
    for i, h in enumerate(hits[:max_hits], 1):
        passes = ", ".join(h.get("passes", []))
        print(f"\n{'='*60}")
        print(f"#{i} | {h['id']} ({h['testament']}) | dist: {h['distance']:.4f} | {passes}")
        print(f"{'='*60}")
        text = h["text"]
        print(text[:600] + ("..." if len(text) > 600 else ""))


# ---------------------------------------------------------------------------
# LLM Synthesis
# ---------------------------------------------------------------------------

def ask_ollama(model: str, system: str, context: str, question: str, stream: bool = True):
    """Send to Ollama chat API and stream the response."""
    try:
        import ollama
    except ImportError:
        print("ERROR: ollama not installed.\n  pip install ollama", file=sys.stderr)
        sys.exit(1)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context}\n\n=== QUESTION ===\n{question}"},
    ]

    if stream:
        response = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in response:
            content = chunk.get("message", {}).get("content", "")
            print(content, end="", flush=True)
        print()  # newline after stream ends
    else:
        response = ollama.chat(model=model, messages=messages, stream=False)
        print(response["message"]["content"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bible Study Oracle — multi-pass retrieval + local LLM synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oracle.py "What does it mean that God rested on the seventh day?"
  oracle.py "What is the heart of man?" --model qwen2.5:14b
  oracle.py "Explain atonement" --n 12 --testament OT
  oracle.py "Who is the Logos?" --raw
  oracle.py "Fear of the Lord" --no-cross
        """
    )
    parser.add_argument("question", help="Biblical question to answer")
    parser.add_argument("--model", default="qwen3.5:9b",
                        help="Ollama model for synthesis (default: qwen3.5:9b)")
    parser.add_argument("--n", type=int, default=7,
                        help="Primary retrieval count (default: 7)")
    parser.add_argument("--cross-n", type=int, default=5,
                        help="Cross-reference retrieval count per pass (default: 5)")
    parser.add_argument("--testament", choices=["OT", "NT"],
                        help="Restrict primary search to one testament")
    parser.add_argument("--no-cross", action="store_true",
                        help="Skip cross-reference passes (Pass 2 + 3)")
    parser.add_argument("--raw", action="store_true",
                        help="Print raw retrieval results, skip LLM synthesis")
    parser.add_argument("--max-context", type=int, default=12,
                        help="Max passages to feed to LLM (default: 12)")
    args = parser.parse_args()

    print(f"\n[Levi] Retrieving passages for: \"{args.question}\"")
    if not args.no_cross:
        print("[Levi] Running 3-pass retrieval (primary + Strong's + cross-testament)...")
    else:
        print("[Levi] Running single-pass retrieval...")

    hits = multi_pass_retrieve(
        question=args.question,
        n_primary=args.n,
        n_cross=args.cross_n,
        testament=args.testament,
        do_cross=not args.no_cross,
    )

    print(f"[Levi] Retrieved {len(hits)} passages")

    if args.raw:
        print_raw(hits, args.max_context)
        return

    context = format_context(hits, args.max_context)

    print(f"\n[Oracle] Synthesizing with {args.model}...\n")
    print("=" * 60)
    ask_ollama(args.model, SYSTEM_PROMPT, context, args.question)
    print("=" * 60)

    # Print citation summary
    primary = [h["id"] for h in hits[:args.max_context] if "primary" in h.get("passes", [])]
    cross = [h["id"] for h in hits[:args.max_context] if "cross-testament" in h.get("passes", [])]
    if primary:
        print(f"\nPrimary: {', '.join(primary[:5])}")
    if cross:
        print(f"Cross-ref: {', '.join(cross[:5])}")


if __name__ == "__main__":
    main()
