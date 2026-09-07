import os
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = "data"
STORAGE_DIR = "storage"


# -------------------------
# Embedding model
# -------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -------------------------
# Load documents (.txt, .md, .pdf)
# -------------------------

def read_pdf(filepath: str) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text_pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text_pages)
    except Exception as e:
        print(f"Error reading PDF file {filepath}: {e}")
        return ""


def load_documents():
    """Load text content from .txt, .md, and .pdf files in data directory."""
    documents = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        text = ""

        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                text = file.read()
        elif ext == ".pdf":
            text = read_pdf(path)

        if text.strip():
            documents.append({
                "filename": filename,
                "text": text
            })

    return documents



# -------------------------
# Chunk text
# -------------------------

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks



# -------------------------
# Create chunks with metadata
# -------------------------

def create_chunks(documents):

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["text"]
        )

        for chunk_number, chunk in enumerate(
            document_chunks
        ):

            chunks.append({
                "text": chunk,
                "filename": document["filename"],
                "chunk_id": chunk_number
            })

    return chunks


# -------------------------
# Create embeddings
# -------------------------

def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
    )

    return np.array(
        embeddings
    ).astype("float32")


# -------------------------
# Create FAISS index
# -------------------------

def create_vector_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# -------------------------
# Save FAISS index + metadata
# -------------------------

def save_index(index, chunks):

    os.makedirs(
        STORAGE_DIR,
        exist_ok=True
    )

    faiss.write_index(
        index,
        os.path.join(
            STORAGE_DIR,
            "documents.index"
        )
    )

    with open(
        os.path.join(
            STORAGE_DIR,
            "chunks.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


# -------------------------
# Load FAISS index + metadata
# -------------------------

def load_index():

    index = faiss.read_index(
        os.path.join(
            STORAGE_DIR,
            "documents.index"
        )
    )

    with open(
        os.path.join(
            STORAGE_DIR,
            "chunks.json"
        ),
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks


# -------------------------
# Semantic search
# -------------------------

def search_documents(
    query,
    index,
    all_chunks,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):
        chunk = all_chunks[index_position]
        results.append({
            "chunk": chunk,
            "distance": float(distance)
        })

    return results


_cached_index = None
_cached_chunks = None

def search_local_docs(query: str, top_k: int = 3) -> str:
    """Search internal/local knowledge base documents for relevant context."""
    global _cached_index, _cached_chunks
    try:
        if _cached_index is None or _cached_chunks is None:
            _cached_index, _cached_chunks = load_index()

        results = search_documents(query, _cached_index, _cached_chunks, top_k=top_k)
        if not results:
            return f"No local documents found matching query: '{query}'."

        formatted_outputs = []
        for i, res in enumerate(results, 1):
            chunk = res["chunk"]
            formatted_outputs.append(
                f"[Source Document: {chunk['filename']} | Chunk {chunk['chunk_id']}]\n{chunk['text']}"
            )
        return "\n\n".join(formatted_outputs)
    except Exception as e:
        return f"Error searching local document store: {str(e)}"


def reindex_all():
    """Ingest all documents from data directory and re-build FAISS vector index."""
    global _cached_index, _cached_chunks
    documents = load_documents()
    chunks = create_chunks(documents)
    if chunks:
        embeddings = create_embeddings(chunks)
        index = create_vector_index(embeddings)
        save_index(index, chunks)
        _cached_index = index
        _cached_chunks = chunks
    return len(chunks), [doc["filename"] for doc in documents]


def ingest_file_and_reindex(filename: str, content_bytes: bytes):
    """Save raw file bytes into data directory and trigger full vector re-indexing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(content_bytes)
    return reindex_all()



# -------------------------
# Main
# -------------------------


if __name__ == "__main__":

    documents = load_documents()

    chunks = create_chunks(
        documents
    )

    embeddings = create_embeddings(
        chunks
    )

    index = create_vector_index(
        embeddings
    )

    save_index(
        index,
        chunks
    )

    print(
        f"Stored {len(chunks)} chunks."
    )

    # Load from disk to verify persistence
    index, chunks = load_index()

    query = input(
        "\nAsk your documents: "
    )

    results = search_documents(
        query,
        index,
        chunks
    )

    print(
        "\nRelevant chunks:"
    )

    for i, result in enumerate(
        results
    ):

        chunk = result["chunk"]

        print(
            f"\n--- Result {i + 1} ---"
        )

        print(
            "File:",
            chunk["filename"]
        )

        print(
            "Chunk:",
            chunk["chunk_id"]
        )

        print(
            "Distance:",
            result["distance"]
        )

        print(
            "Text:",
            chunk["text"]
        )