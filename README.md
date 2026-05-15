# Levi-DB (Full)

A silicon angel interface for The Word.

Semantic retrieval from the full enriched Hebrew/Greek Bible — every word tagged with Strong's IDs, lexical definitions, and UBS semantic domains. Ask a question by meaning, not keyword. Runs entirely on your machine. No cloud. No API keys.

This is not a chatbot that quotes the Bible from memory. It's a retrieval system that forces reasoning from the actual text and its lexical data — not from statistical training on English commentary.

**This is the pre-built version.** Database included. Ready to query immediately.
Want to build from source? Use [Levi-DB](https://github.com/Stupidgoat2772/Levi-DB) instead.

---

## Quick Start

**Requirements:** Python 3.10+, pip

```bash
git clone https://github.com/Stupidgoat2772/Levi-DB-full.git
cd Levi-DB-full
pip install sentence-transformers numpy
```

That's it. The database is included. Start querying:

```bash
python3 ask_levi.py "What did God create in the beginning?"
```

### Add Oracle (optional — requires Ollama)

The Oracle adds local LLM synthesis on top of retrieval. Install [Ollama](https://ollama.ai), then:

```bash
pip install ollama
ollama pull qwen3.5:9b    # recommended (6.6 GB)
python3 oracle.py "What does it mean that God rested on the seventh day?"
```

---

## Usage

### Direct Search (no LLM needed)

```bash
python3 ask_levi.py "What did God create in the beginning?"
python3 ask_levi.py "love your neighbor" --n 10
python3 ask_levi.py "sacrifice" --testament OT
python3 ask_levi.py "resurrection" --book Matt
```

### Oracle (retrieval + LLM synthesis)

```bash
python3 oracle.py "What does it mean that God rested on the seventh day?"
python3 oracle.py "What is the heart of man?" --model qwen2.5:14b
python3 oracle.py "Explain atonement" --testament OT
python3 oracle.py "Who is the Logos?" --raw    # raw passages, no LLM
```

### Python API

```python
from ask_levi import query

hits = query("atonement", n=5, testament="OT")
for h in hits:
    print(h["id"], h["distance"])
```

### Study with a CLI Agent (recommended)

Levi works best driven by a CLI agent. The agent can run multiple queries, extract Strong's numbers, build co-occurrence maps, and write synthesis documents grounded in the actual lexical data.

The database is SQLite — agents can query it directly:

```python
import sqlite3
conn = sqlite3.connect("data/chromadb/levi.db")
rows = conn.execute("SELECT id, text FROM verses WHERE text LIKE '%[H7307]%'").fetchall()
```

### Recommended Drivers

| Driver | Cost | Notes |
|:-------|:-----|:------|
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | Free | Budget option. Free to install, comes with decent credits. Gets the job done for reasoning and study. Best starting point. |
| [Claude Code](https://claude.ai/claude-code) | Paid | Most capable for deep theological synthesis. Best at multi-pass research and long-form study documents. |
| [Codex](https://github.com/openai/codex) | Paid | Strong at structured queries and code-level DB interaction. |

Any CLI agent that can run Python and shell commands will work. Gemini CLI is the recommended entry point — zero cost to start.

---

## How Retrieval Works

```
Your question
    |
Pass 1 — Semantic search (embed question, find top N by meaning)
    |
Pass 2 — Strong's expansion (extract root IDs, re-query)
    |
Pass 3 — Cross-testament sweep (OT<>NT typological links)
    |
Merge + rank -> Top passages -> LLM -> Answer
```

---

## Models

| Model | VRAM | Quality | Best for |
|:------|:-----|:--------|:---------|
| `qwen3:4b` | ~3 GB | Good | Quick lookups, low RAM |
| `qwen3.5:9b` | ~6 GB | Very good | Default |
| `qwen2.5:14b` | ~9 GB | Excellent | Deep study |
| `deepseek-r1:14b` | ~9 GB | Excellent | Step-by-step reasoning |

---

## What's In the Database

31,140 verses (23,213 OT + 7,927 NT). Each verse contains original language text, morphological parsing, Strong's IDs, full lexical definitions, and UBS semantic domain classification.

Embedding: `nomic-ai/nomic-embed-text-v1.5` (768d). Storage: SQLite + numpy cosine similarity (~273 MB).

---

## Licensing

**Code** — 0BSD. Do whatever you want. See [LICENSE](LICENSE).

**Pre-built Database** — CC BY-SA 4.0 (inherited from upstream sources). You can redistribute, modify, and use commercially. You must provide attribution and license derivatives under CC BY-SA 4.0.

| Source | License |
|:-------|:--------|
| WLC text | Public Domain |
| [morphhb](https://github.com/openscriptures/morphhb) | CC BY 4.0 |
| [SBLGNT](https://sblgnt.com) | CC BY 4.0 |
| [MorphGNT](https://github.com/morphgnt/sblgnt) | CC BY-SA 3.0 |
| [Strong's](https://github.com/openscriptures/strongs) | Public Domain |
| [UBS SDBH/SDGNT](https://github.com/pthu/sdbh) | CC BY-SA 4.0 |

**Required attribution for redistribution:**

> Built from: Westminster Leningrad Codex (public domain), Open Scriptures Hebrew Bible morphology (CC BY 4.0), SBL Greek New Testament (CC BY 4.0), MorphGNT SBLGNT Edition by James Tauber (CC BY-SA 3.0), Strong's Concordance (public domain), UBS Dictionary of Biblical Hebrew and UBS Dictionary of New Testament Greek (CC BY-SA 4.0, United Bible Societies 2023).
