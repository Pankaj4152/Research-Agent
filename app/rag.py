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
# Load documents
# -------------------------

def load_documents():

    documents = []

    for filename in os.listdir(DATA_DIR):

        if filename.endswith(".txt"):

            path = os.path.join(
                DATA_DIR,
                filename
            )

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


# -------------------------
# Chunk text
# -------------------------

def chunk_text(text, chunk_size=50):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

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