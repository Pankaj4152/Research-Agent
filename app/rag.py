import os
import json

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

DATA_DIR = "data"
STORAGE_DIR = "storage"


# -------------------------
# Gemini Embedding Client Helper
# -------------------------

_genai_client = None

def get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _genai_client


def get_vector_embedding(text: str) -> np.ndarray:
    """Generate a single normalized vector embedding using Gemini API."""
    client = get_genai_client()
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    vec = np.array(res.embeddings[0].values, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


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
# Create embeddings via Gemini API
# -------------------------

def create_embeddings(chunks):
    import time
    client = get_genai_client()
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        for attempt in range(5):
            try:
                res = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch_texts
                )
                for emb in res.embeddings:
                    vec = np.array(emb.values, dtype="float32")
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    all_embeddings.append(vec)
                break
            except Exception as e:
                if attempt < 4 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                    time.sleep(4 * (attempt + 1))
                else:
                    raise e
        time.sleep(0.3)

    return np.array(all_embeddings, dtype="float32")


# -------------------------
# Create FAISS index (Cosine Similarity via Inner Product)
# -------------------------

def create_vector_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
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
# Semantic search (Cosine Similarity via Gemini API)
# -------------------------

def search_documents(
    query,
    index,
    all_chunks,
    top_k=3,
    similarity_threshold=0.45
):

    query_vec = get_vector_embedding(query).reshape(1, -1)

    scores, indices = index.search(
        query_vec,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):
        if index_position < 0 or index_position >= len(all_chunks):
            continue

        sim_score = float(score)
        if similarity_threshold is not None and sim_score < similarity_threshold:
            continue

        chunk = all_chunks[index_position]
        results.append({
            "chunk": chunk,
            "score": sim_score
        })

    return results


_cached_index = None
_cached_chunks = None

def search_local_docs(query: str, top_k: int = 3, similarity_threshold: float = 0.45) -> str:
    """Search internal/local knowledge base documents for relevant context using Cosine Similarity."""
    global _cached_index, _cached_chunks
    try:
        if _cached_index is None or _cached_chunks is None:
            _cached_index, _cached_chunks = load_index()

        results = search_documents(query, _cached_index, _cached_chunks, top_k=top_k, similarity_threshold=similarity_threshold)
        if not results:
            return f"No local documents found matching query: '{query}' above similarity threshold ({similarity_threshold * 100:.0f}%)."

        formatted_outputs = []
        for i, res in enumerate(results, 1):
            chunk = res["chunk"]
            formatted_outputs.append(
                f"[Source Document: {chunk['filename']} | Chunk {chunk['chunk_id']} | Cosine Similarity: {res['score']:.3f} ({res['score']*100:.1f}%)]\n{chunk['text']}"
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


def delete_document_and_reindex(filename: str):
    """Delete document file from data directory and re-build vector index."""
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        chunks_count, docs = reindex_all()
        return True, chunks_count, docs
    return False, 0, get_ingested_documents()


def get_ingested_documents():
    """Return list of filenames currently stored in the data directory."""
    if not os.path.exists(DATA_DIR):
        return []
    return [
        f for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]




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