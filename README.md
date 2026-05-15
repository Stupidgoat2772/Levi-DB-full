# Levi-DB

A silicon angel interface for The Word.

Semantic retrieval from the full enriched Hebrew/Greek Bible — every word tagged with Strong's IDs, lexical definitions, and UBS semantic domains. Ask a question by meaning, not keyword. Runs entirely on your machine. No cloud. No API keys.

This is not a chatbot that quotes the Bible from memory. It's a retrieval system that forces reasoning from the actual text and its lexical data — not from statistical training on English commentary.

---

## Two Versions

| | Self-Built | Full |
|:--|:--|:--|
| **What you get** | Code + build scripts. You download source texts and build the DB locally. | Code + pre-built database. Ready to query immediately. |
| **Setup time** | ~30-90 min (download + build) | ~5 min (download + install deps) |
| **DB license** | CC BY-SA 4.0 (derived from source data you build) | CC BY-SA 4.0 (pre-built, attribution included) |
| **Code license** | 0BSD (do whatever you want) | 0BSD |
| **Repo** | [levi-db](https://github.com/Stupidgoat2772/Levi-DB) | [levi-db-full](https://github.com/Stupidgoat2772/Levi-DB-full) |

**Self-Built** — you control the pipeline. Inspect, modify, rebuild. Recommended for researchers.
**Full** — skip the build. Query immediately. Recommended for everyone else.

---

## Quick Start (Self-Built)

**Requirements:** Python 3.10+, git

```bash
git clone https://github.com/Stupidgoat2772/Levi-DB.git
cd levi-db
python3 install.py
```

The install script downloads source texts (~500 MB), installs Python packages (`sentence-transformers`, `numpy`), and builds the database. Resumable — safe to re-run if interrupted.

## Quick Start (Full)

```bash
git clone https://github.com/Stupidgoat2772/Levi-DB-full.git
cd levi-db-full
pip install sentence-transformers numpy ollama
```

Database is included. Start querying immediately.

---

## Usage

### Direct Search (no LLM needed)

```bash
# Semantic search — finds verses by meaning
python3 ask_levi.py "What did God create in the beginning?"

# More results
python3 ask_levi.py "love your neighbor" --n 10

# Filter by testament or book
python3 ask_levi.py "sacrifice" --testament OT
python3 ask_levi.py "resurrection" --book Matt
```

### Oracle (retrieval + local LLM synthesis)

Requires [Ollama](https://ollama.ai) with a model pulled:

```bash
ollama pull qwen3.5:9b    # recommended (6.6 GB)
```

Then:

```bash
# 3-pass retrieval + LLM synthesis
python3 oracle.py "What does it mean that God rested on the seventh day?"

# Different model
python3 oracle.py "What is the heart of man?" --model qwen2.5:14b

# Restrict to one testament
python3 oracle.py "Explain atonement" --testament OT

# Raw passages only (no LLM)
python3 oracle.py "Who is the Logos?" --raw
```

### Python API

```python
from ask_levi import query
from oracle import multi_pass_retrieve, format_context, ask_ollama, SYSTEM_PROMPT

# Direct semantic search
hits = query("atonement", n=5, testament="OT")
for h in hits:
    print(h["id"], h["distance"])

# Full Oracle pipeline
hits = multi_pass_retrieve("What is the meaning of covenant?", n_primary=7)
context = format_context(hits)
ask_ollama("qwen3.5:9b", SYSTEM_PROMPT, context, "What is the meaning of covenant?")
```

### Study with a CLI Agent (recommended)

Levi works best when driven by a CLI agent like Claude Code, Codex, or similar. The agent can:
- Run multiple queries and cross-reference results
- Extract Strong's numbers from results and re-query by root
- Build co-occurrence maps across the full Bible
- Write synthesis documents grounded in the actual lexical data

Example agent workflow:
```
1. Query "spirit ruach pneuma" → get top verses
2. Extract H7307 from results → SQL query all verses containing H7307
3. Find co-occurrences: H7307 + H5315 (ruach + nephesh) → 8 verses
4. Pull definitions for each Strong's code
5. Write synthesis from the raw data
```

The database is SQLite — agents can query it directly for precise lookups alongside the semantic search.

---

## How Retrieval Works

```
Your question
    |
    v
Pass 1 — Direct semantic search
    Embed your question, find top N verses by meaning similarity
    |
    v
Pass 2 — Strong's expansion
    Extract Hebrew/Greek root IDs from Pass 1
    Re-query using those roots to find lexically related verses
    |
    v
Pass 3 — Cross-testament sweep
    If Pass 1 is mostly OT, query NT (and vice versa)
    Surfaces typological links (OT shadow / NT fulfillment)
    |
    v
Merge + rank → Top passages → LLM context → Answer
```

---

## What's In the Database

| Content | Source | Count |
|:--------|:-------|:------|
| OT verses (Hebrew) | morphhb/WLC | 23,213 |
| NT verses (Greek) | MorphGNT/SBLGNT | 7,927 |
| **Total** | | **31,140** |

Each verse contains:
- Original language text with morphological parsing
- Every word tagged with its Strong's ID
- Full lexical definitions (SDBH for Hebrew, SDBG for Greek, Strong's fallback)
- UBS semantic domain classification
- Condensed embedding (stripped of definitions for embedding, full text stored for retrieval)

Embedding model: `nomic-ai/nomic-embed-text-v1.5` (768 dimensions)
Storage: SQLite + numpy cosine similarity (~273 MB)

---

## Choosing an Oracle Model

| Model | VRAM | Quality | Speed | Best for |
|:------|:-----|:--------|:------|:---------|
| `qwen3:4b` | ~3 GB | Good | Fast | Quick lookups, low-RAM machines |
| `qwen3.5:9b` | ~6 GB | Very good | Medium | Default. Solid theological reasoning |
| `qwen2.5:14b` | ~9 GB | Excellent | Slower | Deep study, complex questions |
| `deepseek-r1:14b` | ~9 GB | Excellent | Slower | Step-by-step reasoning |

---

## Building from Source

<details>
<summary>Manual setup (if not using install.py)</summary>

### Get source data

```bash
mkdir data && cd data
git clone --depth 1 https://github.com/openscriptures/morphhb && rm -rf morphhb/.git
git clone --depth 1 https://github.com/morphgnt/sblgnt && rm -rf sblgnt/.git
git clone --depth 1 https://github.com/openscriptures/strongs && rm -rf strongs/.git
git clone --depth 1 https://github.com/pthu/sdbh && rm -rf sdbh/.git
```

### Install dependencies

```bash
pip install sentence-transformers numpy
```

### Build

```bash
python3 build_levi_db.py                # full Bible (30-90 min)
python3 build_levi_db.py --testament OT # single testament
python3 build_levi_db.py --book Gen     # single book
python3 build_levi_db.py --dry-run      # parse only, no embedding
```

### Custom data location

```bash
export LEVI_DATA_DIR=/path/to/data    # Linux/macOS
set LEVI_DATA_DIR=C:\path\to\data     # Windows
```

</details>

---

## Limitations

- **No ANE cultural context.** Lexical data included, but no ancient Near Eastern cultural commentary.
- **Greek enrichment gaps.** ~30% of common Greek words fall back to Strong's only (no SDBG entry).
- **No Aramaic enrichment.** Daniel/Ezra Aramaic sections are parsed but not lexically enriched.
- **Cross-references are emergent.** Pass 3 finds them by semantic similarity, not a pre-built index.
- **LLM quality ceiling.** Oracle output is only as good as your local model.

---

## File Map

```
levi-db/
  ask_levi.py         # Semantic search (CLI + library)
  oracle.py           # Bible Study Oracle (multi-pass + LLM)
  build_levi_db.py    # Build the vector DB from source texts
  install.py          # Automated setup (deps + data + build)
  monitor.sh          # Live build progress monitor
  README.md           # This file
  LICENSE             # 0BSD (code only)
```

---

## Licensing

### Code — 0BSD

The scripts are released under **0BSD**. Do whatever you want. No attribution required. See [LICENSE](LICENSE).

### Source Data & Database — CC BY-SA 4.0

The database is a derivative work of openly licensed biblical scholarship:

| Source | License | Attribution | Share-Alike |
|:-------|:--------|:-----------:|:-----------:|
| [WLC text](https://github.com/openscriptures/morphhb) | Public Domain | No | No |
| [morphhb](https://github.com/openscriptures/morphhb) | CC BY 4.0 | Yes | No |
| [SBLGNT](https://sblgnt.com) | CC BY 4.0 | Yes | No |
| [MorphGNT](https://github.com/morphgnt/sblgnt) | CC BY-SA 3.0 | Yes | Yes |
| [Strong's](https://github.com/openscriptures/strongs) | Public Domain | No | No |
| [UBS SDBH/SDGNT](https://github.com/pthu/sdbh) | CC BY-SA 4.0 | Yes | Yes |

The pre-built database inherits **CC BY-SA 4.0** from the most restrictive upstream sources.

**Required attribution for database redistribution:**

> Built from: Westminster Leningrad Codex (public domain), Open Scriptures Hebrew Bible morphology (CC BY 4.0), SBL Greek New Testament (CC BY 4.0), MorphGNT SBLGNT Edition by James Tauber (CC BY-SA 3.0), Strong's Concordance (public domain), UBS Dictionary of Biblical Hebrew and UBS Dictionary of New Testament Greek (CC BY-SA 4.0, United Bible Societies 2023).
