#!/usr/bin/env python3
"""build_levi_db.py — Phase 1: Build base Levi vector DB chunks.

Parses morphhb (OT) and SBLGNT (NT), enriches each word with Strong's
definitions and SDBH semantic domains, compiles into Sof Pasuq / verse
bounded chunks, and stores in ChromaDB.

Usage:
    build_levi_db.py                    # build full DB
    build_levi_db.py --book Gen         # build single book
    build_levi_db.py --testament OT     # build OT only
    build_levi_db.py --testament NT     # build NT only
    build_levi_db.py --dry-run          # print chunks, don't embed
"""

import json
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(os.environ.get("LEVI_DATA_DIR", Path(__file__).resolve().parent / "data"))
DATA = ROOT
MORPHHB_DIR = DATA / "morphhb" / "wlc"
SBLGNT_DIR = DATA / "sblgnt"
STRONGS_HEB = DATA / "strongs" / "hebrew" / "strongs-hebrew-dictionary.js"
STRONGS_GRK = DATA / "strongs" / "greek" / "strongs-greek-dictionary.js"
SDBH_HEB = DATA / "sdbh" / "dictionaries" / "hebrew" / "JSON" / "UBSHebrewDic-v0.9.2-en.JSON"
SDBH_GRK = DATA / "sdbh" / "dictionaries" / "greek" / "JSON" / "UBSGreekNTDic-v1.1-en.JSON"
SDBH_DOMAINS = DATA / "sdbh" / "dictionaries" / "hebrew" / "JSON" / "UBSHebrewDicLexicalDomains-v0.9.2-en.JSON"
DB_DIR = DATA / "chromadb"

# morphhb OSIS namespace
OSIS_NS = "http://www.bibletechnologies.net/2003/OSIS/namespace"

# OT book order (filename stems in morphhb)
OT_BOOKS = [
    "Gen", "Exod", "Lev", "Num", "Deut", "Josh", "Judg", "Ruth",
    "1Sam", "2Sam", "1Kgs", "2Kgs", "1Chr", "2Chr", "Ezra", "Neh",
    "Esth", "Job", "Ps", "Prov", "Eccl", "Song", "Isa", "Jer",
    "Lam", "Ezek", "Dan", "Hos", "Joel", "Amos", "Obad", "Jonah",
    "Mic", "Nah", "Hab", "Zeph", "Hag", "Zech", "Mal",
]

# NT book files in SBLGNT (number-name pairs)
NT_BOOKS = [
    "61-Mt", "62-Mk", "63-Lk", "64-Jn", "65-Ac",
    "66-Ro", "67-1Co", "68-2Co", "69-Ga", "70-Eph",
    "71-Php", "72-Col", "73-1Th", "74-2Th", "75-1Ti",
    "76-2Ti", "77-Tit", "78-Phm", "79-Heb", "80-Jas",
    "81-1Pe", "82-2Pe", "83-1Jn", "84-2Jn", "85-3Jn",
    "86-Jud", "87-Re",
]


def strip_accents(s: str) -> str:
    """Normalize Unicode and strip combining accents for matching."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def condense_for_embedding(text: str) -> str:
    """Strip definitions and domain labels, keep verse ref + words + Strong's codes.

    The full text is stored in the DB; this condensed version is only used
    to generate the embedding vector. Handles nested parentheses in SDBH defs.
    """
    # Strip balanced parentheses (handles nesting) and [Domain: ...] tags
    # Walk char by char to handle nested parens correctly
    result = []
    depth = 0
    i = 0
    while i < len(text):
        # Skip [Domain: ...] tags
        if text[i] == '[' and text[i:i+8] == '[Domain:':
            j = text.find(']', i)
            if j != -1:
                i = j + 1
                continue
        if text[i] == '(':
            depth += 1
            i += 1
            continue
        if text[i] == ')':
            if depth > 0:
                depth -= 1
            i += 1
            continue
        if depth == 0:
            result.append(text[i])
        i += 1
    condensed = ''.join(result)
    # Collapse whitespace
    condensed = re.sub(r'  +', ' ', condensed)
    condensed = re.sub(r' \|', '|', condensed)
    return condensed.strip()


# ── Loaders ──────────────────────────────────────────────────────────

def load_strongs(path: Path) -> dict:
    """Parse Strong's JS dictionary file into {H0001: {...}} dict."""
    text = path.read_text(encoding="utf-8")
    # Extract the JSON object from the JS variable assignment
    # Format: var strongsHebrewDictionary = {"H1":{...},...};
    start = text.find("{")
    if start == -1:
        print(f"WARN: Could not parse {path.name}")
        return {}
    # Find matching end — it's the last }
    end = text.rfind("}")
    raw = json.loads(text[start:end + 1])
    # Normalize keys to zero-padded 4-digit format
    out = {}
    for k, v in raw.items():
        out[k] = v
    return out


def load_sdbh(path: Path) -> dict:
    """Load SDBH JSON into {strong_code: entry} lookup."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for entry in data:
        for code in entry.get("StrongCodes", []):
            lookup[code] = entry
    return lookup


def load_sdbh_with_lemma_index(path: Path) -> tuple[dict, dict]:
    """Load SDBH with both Strong's and lemma lookups."""
    data = json.loads(path.read_text(encoding="utf-8"))
    by_strong = {}
    by_lemma = {}
    for entry in data:
        for code in entry.get("StrongCodes", []):
            by_strong[code] = entry
        lemma = entry.get("Lemma", "")
        if lemma:
            by_lemma[lemma] = entry
            by_lemma[strip_accents(lemma)] = entry
    return by_strong, by_lemma


def load_domains(path: Path) -> dict:
    """Load lexical domains into {code: label} lookup."""
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for d in data:
        code = d.get("Code", "")
        labels = d.get("SemanticDomainLocalizations", [])
        if labels:
            lookup[code] = labels[0].get("Label", "")
    return lookup


# ── Enrichment ───────────────────────────────────────────────────────

def get_strongs_def(strongs_dict: dict, code: str) -> str:
    """Get definition from Strong's for a given code like H7225."""
    entry = strongs_dict.get(code)
    if not entry:
        return ""
    parts = []
    sdef = entry.get("strongs_def", "")
    if sdef:
        parts.append(sdef.strip("{}"))
    kjv = entry.get("kjv_def", "")
    if kjv:
        parts.append(kjv)
    return "; ".join(parts)


def get_sdbh_info(sdbh_dict: dict, code: str) -> tuple[str, list[str]]:
    """Get SDBH definition + domain list for a Strong's code."""
    entry = sdbh_dict.get(code)
    if not entry:
        return "", []

    definitions = []
    domains = []

    for bf in entry.get("BaseForms", []):
        for meaning in bf.get("LEXMeanings", []):
            # Domains
            for dom in (meaning.get("LEXDomains") or []):
                d = dom.get("Domain", "")
                if d and d not in domains:
                    domains.append(d)
            # Senses
            for sense in meaning.get("LEXSenses", []):
                glosses = sense.get("Glosses", [])
                short_def = sense.get("DefinitionShort", "")
                if short_def:
                    definitions.append(short_def)
                elif glosses:
                    definitions.append(", ".join(glosses))

    return "; ".join(definitions[:3]), domains


# ── OT Parser (morphhb XML) ─────────────────────────────────────────

def parse_ot_book(xml_path: Path, strongs_heb: dict, sdbh_heb: dict) -> list[dict]:
    """Parse a morphhb XML file into enriched verse chunks."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chunks = []

    for verse in root.iter(f"{{{OSIS_NS}}}verse"):
        osis_id = verse.attrib.get("osisID", "")
        if not osis_id:
            continue

        words = []
        for w in verse.iter(f"{{{OSIS_NS}}}w"):
            hebrew = w.text or ""
            lemma_raw = w.attrib.get("lemma", "")
            morph = w.attrib.get("morph", "")

            # Parse lemma — can be "b/7225" (prefix/strong) or "1254 a"
            enriched_parts = []
            for part in lemma_raw.split("/"):
                part = part.strip()
                # Skip prefix markers (single letters)
                if len(part) <= 2 and not part.isdigit():
                    continue
                # Extract Strong's number
                num = re.match(r"(\d+)", part)
                if not num:
                    continue
                strong_code = f"H{num.group(1).zfill(4)}"

                strongs_def = get_strongs_def(strongs_heb, strong_code)
                sdbh_def, sdbh_domains = get_sdbh_info(sdbh_heb, strong_code)

                word_entry = f"{hebrew} [{strong_code}]"
                defs = sdbh_def or strongs_def
                if defs:
                    word_entry += f" ({defs})"
                if sdbh_domains:
                    word_entry += f" [Domain: {', '.join(sdbh_domains)}]"

                enriched_parts.append(word_entry)

            if enriched_parts:
                words.append(" ".join(enriched_parts))
            elif hebrew:
                words.append(hebrew)

        if words:
            chunk_text = f"[{osis_id}] " + " | ".join(words)
            chunks.append({
                "id": osis_id,
                "text": chunk_text,
                "testament": "OT",
                "book": osis_id.split(".")[0],
            })

    return chunks


# ── NT Parser (SBLGNT text) ─────────────────────────────────────────

def parse_nt_book(txt_path: Path, strongs_grk: dict, sdbh_grk: dict,
                  sdbh_grk_lemma: dict, book_name: str) -> list[dict]:
    """Parse a SBLGNT morphgnt file into enriched verse chunks."""
    chunks = []
    current_ref = None
    current_words = []

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        ref_code = parts[0]  # e.g., "010201" = book01 ch02 v01
        # pos = parts[1]
        # morph = parts[2]
        text_form = parts[3]
        # normalized = parts[4]
        lemma = parts[6] if len(parts) > 6 else parts[5]

        # Ref format: BBCCVV (book, chapter, verse)
        verse_ref = ref_code[2:6]  # CCVV
        ch = int(ref_code[2:4])
        vs = int(ref_code[4:6])
        osis_id = f"{book_name}.{ch}.{vs}"

        if verse_ref != current_ref:
            # Flush previous verse
            if current_ref and current_words:
                prev_ch = int(current_ref[:2])
                prev_vs = int(current_ref[2:])
                prev_id = f"{book_name}.{prev_ch}.{prev_vs}"
                chunk_text = f"[{prev_id}] " + " | ".join(current_words)
                chunks.append({
                    "id": prev_id,
                    "text": chunk_text,
                    "testament": "NT",
                    "book": book_name,
                })
            current_ref = verse_ref
            current_words = []

        # Try to find Strong's code via SDBH lemma lookup
        sdbh_entry = sdbh_grk_lemma.get(lemma) or sdbh_grk_lemma.get(strip_accents(lemma))
        strong_code = ""
        if sdbh_entry:
            codes = sdbh_entry.get("StrongCodes", [])
            if codes:
                strong_code = codes[0]

        # Build enriched word
        word_entry = text_form
        if strong_code:
            strongs_def = get_strongs_def(strongs_grk, strong_code)
            sdbh_def, sdbh_domains = get_sdbh_info(sdbh_grk, strong_code)
            word_entry = f"{text_form} [{strong_code}]"
            defs = sdbh_def or strongs_def
            if defs:
                word_entry += f" ({defs})"
            if sdbh_domains:
                word_entry += f" [Domain: {', '.join(sdbh_domains)}]"

        current_words.append(word_entry)

    # Flush last verse
    if current_ref and current_words:
        ch = int(current_ref[:2])
        vs = int(current_ref[2:])
        osis_id = f"{book_name}.{ch}.{vs}"
        chunk_text = f"[{osis_id}] " + " | ".join(current_words)
        chunks.append({
            "id": osis_id,
            "text": chunk_text,
            "testament": "NT",
            "book": book_name,
        })

    return chunks


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")

    target_book = None
    target_testament = None

    if "--book" in args:
        idx = args.index("--book")
        target_book = args[idx + 1]
    if "--testament" in args:
        idx = args.index("--testament")
        target_testament = args[idx + 1].upper()

    # Load dictionaries
    print("Loading Strong's Hebrew...")
    strongs_heb = load_strongs(STRONGS_HEB)
    print(f"  {len(strongs_heb)} entries")

    print("Loading Strong's Greek...")
    strongs_grk = load_strongs(STRONGS_GRK)
    print(f"  {len(strongs_grk)} entries")

    print("Loading SDBH Hebrew...")
    sdbh_heb = load_sdbh(SDBH_HEB)
    print(f"  {len(sdbh_heb)} entries")

    print("Loading SDBH Greek...")
    sdbh_grk, sdbh_grk_lemma = load_sdbh_with_lemma_index(SDBH_GRK)
    print(f"  {len(sdbh_grk)} by Strong's, {len(sdbh_grk_lemma)} by lemma")

    all_chunks = []

    # OT
    if target_testament in (None, "OT"):
        books = [target_book] if target_book and target_book in OT_BOOKS else OT_BOOKS
        if target_book and target_book not in OT_BOOKS and target_testament == "OT":
            print(f"Book {target_book} not found in OT")
            books = []

        for book in books:
            xml_path = MORPHHB_DIR / f"{book}.xml"
            if not xml_path.exists():
                print(f"  WARN: {xml_path} not found, skipping")
                continue
            chunks = parse_ot_book(xml_path, strongs_heb, sdbh_heb)
            all_chunks.extend(chunks)
            print(f"  {book}: {len(chunks)} verses")

    # NT
    if target_testament in (None, "NT"):
        for nt_file in NT_BOOKS:
            book_name = nt_file.split("-", 1)[1]
            if target_book and book_name != target_book:
                continue
            txt_path = SBLGNT_DIR / f"{nt_file}-morphgnt.txt"
            if not txt_path.exists():
                print(f"  WARN: {txt_path} not found, skipping")
                continue
            chunks = parse_nt_book(txt_path, strongs_grk, sdbh_grk,
                                   sdbh_grk_lemma, book_name)
            all_chunks.extend(chunks)
            print(f"  {book_name}: {len(chunks)} verses")

    print(f"\nTotal chunks: {len(all_chunks)}")

    if dry_run:
        # Print first 10 chunks
        for chunk in all_chunks[:10]:
            print(f"\n{'='*60}")
            print(chunk["text"])
        print(f"\n... ({len(all_chunks) - 10} more)")
        return

    # Embed and store in SQLite
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run:")
        print("  pip install sentence-transformers")
        sys.exit(1)

    import sqlite3
    import struct

    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DB_DIR / "levi.db"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""CREATE TABLE IF NOT EXISTS verses (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        testament TEXT NOT NULL,
        book TEXT NOT NULL,
        embedding BLOB NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_testament ON verses(testament)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_book ON verses(book)")
    conn.commit()

    # Load embedding model (uses GPU if available, else CPU)
    import torch
    torch.set_num_threads(4)
    print("Loading embedding model (nomic-embed-text)...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    print(f"  Device: {model.device}")

    # Full rebuild: clear table. Partial: append.
    if target_testament is None and target_book is None:
        conn.execute("DELETE FROM verses")
        conn.commit()

    # Skip already-embedded chunks (enables safe resume)
    existing_ids = set(r[0] for r in conn.execute("SELECT id FROM verses").fetchall())
    all_chunks = [c for c in all_chunks if c["id"] not in existing_ids]
    if existing_ids:
        print(f"  Skipping {len(existing_ids)} already-embedded. {len(all_chunks)} remaining.")

    if not all_chunks:
        print("  Nothing to embed.")
        conn.close()
        total = len(existing_ids)
        print(f"\nDone. {total} verses in {db_path}")
        return

    # Embed and store in batches to avoid OOM on large builds
    BATCH_SIZE = 128
    total_chunks = len(all_chunks)
    print(f"  Embedding {total_chunks} chunks (batch_size={BATCH_SIZE})...")

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        embed_inputs = [condense_for_embedding(c["text"]) for c in batch]
        embeddings = model.encode(embed_inputs, batch_size=BATCH_SIZE)

        for c, emb in zip(batch, embeddings):
            blob = struct.pack(f'{len(emb)}f', *emb)
            conn.execute(
                "INSERT OR REPLACE INTO verses (id, text, testament, book, embedding) VALUES (?, ?, ?, ?, ?)",
                (c["id"], c["text"], c["testament"], c["book"], blob),
            )
        conn.commit()
        done = min(i + len(batch), total_chunks)
        print(f"  {done}/{total_chunks} ({done * 100 // total_chunks}%)")

    conn.close()

    total = len(existing_ids) + total_chunks
    print(f"\nDone. {total} verses in {db_path}")


if __name__ == "__main__":
    main()
