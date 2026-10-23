"""
ingest.py

Loads documents from the ant_docs folder, splits them into chunks,
embeds each chunk using OpenAI's text-embedding-3-small model, and
stores the resulting vectors in a local ChromaDB vector store.

Run this once before using query.py. If you add new documents,
delete the chroma_db folder and re-run to rebuild the vector store.

Usage:
    export OPENAI_API_KEY='your-key-here'
    python3 ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent          # folder this script lives in
DOCS_DIR   = BASE_DIR / "ant_docs"          # folder containing .txt source files
CHROMA_DIR = BASE_DIR / "chroma_db"         # where the vector store will be saved

# ── Config ────────────────────────────────────────────────────────────────────

# Documents to ingest -- start with 5, expand to all 12 once pipeline is verified
DOCS_TO_INGEST = [
    "01_natgeo_ants_facts.txt",
    "02_quanta_algorithms.txt",
    "10_nature_scitable_social_parasitism.txt",
    "11_radiolab_argentine_invasion.txt",
    "12_superorganism_wilson_holldobler.txt",
]

# Chunking settings
# chunk_size: max characters per chunk. Smaller = more precise retrieval,
#             less context per chunk. 500 is a solid starting point.
# chunk_overlap: characters shared between adjacent chunks. Prevents key
#                information from being lost at chunk boundaries.
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# Embedding model
# text-embedding-3-small: fast, cheap, high quality. Industry standard.
EMBEDDING_MODEL = "text-embedding-3-small"

# ── Load ──────────────────────────────────────────────────────────────────────

def load_documents() -> list[Document]:
    """Read each .txt file and wrap it in a LangChain Document object."""
    documents = []

    for filename in DOCS_TO_INGEST:
        filepath = DOCS_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found, skipping.")
            continue

        text = filepath.read_text(encoding="utf-8")

        # Pull the source URL out of the file header for metadata
        source_url = "unknown"
        for line in text.splitlines()[:5]:
            if line.startswith("URL:"):
                source_url = line.replace("URL:", "").strip()
                break

        documents.append(Document(
            page_content=text,
            metadata={
                "source":   filename,
                "url":      source_url,
            }
        ))
        print(f"  Loaded: {filename} ({len(text):,} chars)")

    return documents

# ── Chunk ─────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks.

    RecursiveCharacterTextSplitter splits on paragraph breaks first,
    then sentences, then words -- so chunks stay semantically coherent
    rather than cutting mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks

# ── Embed and store ───────────────────────────────────────────────────────────

def build_vector_store(chunks: list[Document], api_key: str) -> Chroma:
    """
    Embed all chunks and persist them to a local ChromaDB vector store.

    Each chunk's text is sent to the OpenAI embedding API, which returns
    a 1536-dimensional vector representing its meaning. Chunks with similar
    meaning will have vectors that are close together in that space --
    which is what makes semantic search possible at query time.
    """
    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Ant Colony RAG -- Ingestion ===\n")

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Run: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Check docs folder exists
    if not DOCS_DIR.exists():
        print(f"ERROR: ant_docs folder not found at {DOCS_DIR}")
        print("Make sure ant_docs/ is in the same folder as ingest.py")
        sys.exit(1)

    # Load
    print("Loading documents...")
    docs = load_documents()
    if not docs:
        print("ERROR: No documents loaded. Check your ant_docs folder.")
        sys.exit(1)

    # Chunk
    print(f"\nChunking ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)...")
    chunks = chunk_documents(docs)
    print(f"  {len(docs)} documents → {len(chunks)} chunks")
    print(f"  Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    # Embed and store
    print(f"\nEmbedding with {EMBEDDING_MODEL} and storing in ChromaDB...")
    print(f"  This makes one API call to OpenAI -- costs fractions of a cent.")
    build_vector_store(chunks, api_key)

    print(f"\n✓ Vector store saved to ./{CHROMA_DIR.name}/")
    print(f"✓ {len(chunks)} chunks ready to query.")
    print("\nNext step: run query.py to ask questions.")


if __name__ == "__main__":
    main()