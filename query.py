"""
query.py

Loads the vector store built by ingest.py, retrieves the most relevant
chunks for a given question, and passes them to Claude to generate a
grounded answer.

Usage:
    export OPENAI_API_KEY='your-key-here'
    export ANTHROPIC_API_KEY='your-key-here'
    python3 query.py
"""

import os
import sys
from pathlib import Path

import anthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
CLAUDE_MODEL    = "claude-sonnet-4-6"

# How many chunks to retrieve per query.
# More chunks = more context for Claude, but also more noise.
# 4 is a reliable default for a small knowledge base.
TOP_K = 4

# ── Prompt ────────────────────────────────────────────────────────────────────

# The system prompt is where you constrain Claude's behavior.
# "Only use the context below" is the key RAG instruction -- it prevents
# Claude from answering from its general training data instead of your
# documents. This is what makes the system trustworthy and grounded.
SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in ant colony biology.
Answer the user's question using only the context provided below.
If the context does not contain enough information to answer the question,
say so clearly rather than guessing.
Cite which source your answer draws from at the end of your response."""

# ── Load vector store ─────────────────────────────────────────────────────────

def load_vector_store(openai_api_key: str) -> Chroma:
    """Load the existing vector store from disk."""
    if not CHROMA_DIR.exists():
        print("ERROR: chroma_db/ folder not found.")
        print("Run ingest.py first to build the vector store.")
        sys.exit(1)

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=openai_api_key,
    )

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )

    return vector_store

# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve(vector_store: Chroma, question: str, k: int = TOP_K) -> list:
    """
    Embed the question and find the k most semantically similar chunks.

    Returns a list of (Document, score) tuples. Score is cosine similarity:
    1.0 = identical meaning, 0.0 = unrelated. In practice good matches
    score above 0.7 for this kind of factual knowledge base.
    """
    results = vector_store.similarity_search_with_score(question, k=k)
    return results

# ── Build context ─────────────────────────────────────────────────────────────

def build_context(results: list) -> str:
    """
    Format retrieved chunks into a single context string for Claude.

    Each chunk is labeled with its source file so Claude can cite it.
    This is what gets injected into the prompt between the system
    instructions and the user's question.
    """
    context_parts = []
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[Source {i+1}: {source}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)

# ── Ask Claude ────────────────────────────────────────────────────────────────

def ask_claude(question: str, context: str, anthropic_api_key: str) -> str:
    """
    Send the question and retrieved context to Claude and return the answer.

    The full prompt Claude receives looks like:
        System: You are a knowledgeable assistant... [constraints]
        User:   Context: [retrieved chunks]
                Question: [user's question]

    Claude never sees the vector store or knows about embeddings --
    it just gets text and generates a response grounded in that text.
    """
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return response.content[0].text

# ── Main ──────────────────────────────────────────────────────────────────────

def answer(question: str, vector_store: Chroma, anthropic_api_key: str) -> str:
    """Full RAG pipeline: retrieve relevant chunks, then generate an answer."""
    results = retrieve(vector_store, question)
    context = build_context(results)
    return ask_claude(question, context, anthropic_api_key)


def main():
    print("=== Ant Colony RAG -- Query ===\n")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    if not anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print("Loading vector store...")
    vector_store = load_vector_store(openai_api_key)
    print("Ready.\n")

    # Run eval questions to verify the pipeline end to end
    eval_questions = [
        "How do ants communicate with each other?",
        "What are slave-making ants and how do they work?",
        "Why are Argentine ants so successful as invaders?",
        "What is a superorganism?",
        "How do leafcutter ants farm fungus?",
    ]

    for question in eval_questions:
        print(f"Q: {question}")
        print("-" * 60)
        response = answer(question, vector_store, anthropic_api_key)
        print(response)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()