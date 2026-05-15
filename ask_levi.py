#!/usr/bin/env python3
"""ask_levi.py — Query the Levi vector DB.

Semantic search across the enriched Hebrew/Greek scripture database.
Returns top matching verses with their enriched text and metadata.

Usage:
    ask_levi.py "What did God create in the beginning?"
    ask_levi.py "love your neighbor" --n 10
    ask_levi.py "sacrifice" --testament OT
    ask_levi.py "resurrection" --book Matt
"""

import argparse
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("LEVI_DATA_DIR", Path(__file__).resolve().parent / "data"))
DB_PATH = ROOT / "chromadb" / "levi.db"

# Lazy-loaded model (loads once on first query)
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    return _model


def _embed(text: str) -> np.ndarray:
    """Embed a single query string."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True).astype(np.float32)


def _blob_to_vec(blob: bytes) -> np.ndarray:
    """Unpack a float32 blob into a numpy vector."""
    n = len(blob) // 4
    return np.array(struct.unpack(f'{n}f', blob), dtype=np.float32)


def query(text: str, n: int = 5, testament: str = None, book: str = None) -> list[dict]:
    """Query Levi DB. Returns list of {id, text, distance, testament, book}."""
    conn = sqlite3.connect(str(DB_PATH))

    # Build filter
    where = "WHERE 1=1"
    params = []
    if testament:
        where += " AND testament = ?"
        params.append(testament)
    if book:
        where += " AND book = ?"
        params.append(book)

    rows = conn.execute(f"SELECT id, text, testament, book, embedding FROM verses {where}", params).fetchall()
    conn.close()

    if not rows:
        return []

    # Embed query
    q_vec = _embed(text)

    # Cosine similarity against all matching verses
    ids, texts, testaments, books, vecs = [], [], [], [], []
    for row in rows:
        ids.append(row[0])
        texts.append(row[1])
        testaments.append(row[2])
        books.append(row[3])
        vecs.append(_blob_to_vec(row[4]))

    matrix = np.stack(vecs)  # (N, 768)
    # Normalize for cosine similarity
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    similarities = m_norms @ q_norm  # (N,)
    distances = 1 - similarities  # cosine distance (lower = more similar)

    # Top N
    top_idx = np.argsort(distances)[:n]

    hits = []
    for idx in top_idx:
        hits.append({
            "id": ids[idx],
            "text": texts[idx],
            "distance": float(distances[idx]),
            "testament": testaments[idx],
            "book": books[idx],
        })
    return hits


def main():
    parser = argparse.ArgumentParser(description="Query the Levi vector DB")
    parser.add_argument("question", help="Natural language query")
    parser.add_argument("--n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--testament", choices=["OT", "NT"], help="Filter by testament")
    parser.add_argument("--book", help="Filter by book (e.g. Gen, Matt)")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print("ERROR: Levi DB not found. Run build_levi_db.py first.", file=sys.stderr)
        sys.exit(1)

    hits = query(args.question, n=args.n, testament=args.testament, book=args.book)

    if not hits:
        print("No results found.")
        return

    for i, hit in enumerate(hits, 1):
        print(f"\n{'='*60}")
        print(f"#{i} | {hit['id']} ({hit['testament']}) | distance: {hit['distance']:.4f}")
        print(f"{'='*60}")
        text = hit["text"]
        if len(text) > 500:
            print(text[:500] + "...")
            print(f"\n  [{len(text)} chars total]")
        else:
            print(text)


if __name__ == "__main__":
    main()
