import uuid
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.main import ask_agent, ask_agent_stream, clear_session, get_session_history
from app.rag import ingest_file_and_reindex, get_ingested_documents

# Initialize FastAPI app
app = FastAPI(
    title="Research Agent API",
    description="REST API for an Autonomous AI Agent powered by Gemini & Live API Tools",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class ResearchRequest(BaseModel):
    prompt: str
    session_id: str | None = None


# Response schema
class ResearchResponse(BaseModel):
    query: str
    answer: str
    session_id: str
    tool_traces: list[dict] = []


@app.get("/healthz")
@app.get("/healtz")
@app.get("/health")
@app.get("/api/health")
def health_check():
    """Health check endpoint for cloud monitoring (Render, AWS, Docker) and UI telemetry."""
    docs = get_ingested_documents()
    return {
        "status": "online",
        "service": "ResearcheX Core API",
        "version": "1.0.0",
        "active_agents": 3,
        "ingested_documents_count": len(docs),
        "docs_url": "/docs"
    }


@app.get("/api/tools")
def list_tools():
    """List all registered agent tools and capabilities."""
    return {
        "count": 3,
        "tools": [
            {
                "id": "search_local_docs",
                "name": "Local Vector RAG Agent",
                "description": "FAISS Cosine Similarity search over ingested documents (.pdf, .md, .txt)",
                "status": "online",
                "badge": "VECTOR RAG"
            },
            {
                "id": "get_weather",
                "name": "Live Weather Telemetry",
                "description": "Real-time meteorological telemetry lookup via Open-Meteo API",
                "status": "online",
                "badge": "LIVE API"
            },
            {
                "id": "search_wikipedia",
                "name": "Global Knowledge Agent",
                "description": "Encyclopedic & factual research lookup via Wikipedia REST API",
                "status": "online",
                "badge": "GLOBAL DATA"
            }
        ]
    }


@app.post("/api/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    """Execute research prompt via AI Agent and return response with multi-turn session state & tool traces."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        answer, session_id, traces = ask_agent(request.prompt, session_id=request.session_id)
        return ResearchResponse(query=request.prompt, answer=answer, session_id=session_id, tool_traces=traces)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/research/stream")
def research_stream(request: ResearchRequest):
    """Execute research prompt with live Server-Sent Events stream of tool executions and response generation."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    return StreamingResponse(
        ask_agent_stream(request.prompt, session_id=request.session_id),
        media_type="application/x-ndjson"
    )


@app.get("/api/session/{session_id}")
def fetch_session_history(session_id: str):
    """Get chat history for a session."""
    history = get_session_history(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history
    }


@app.delete("/api/session/{session_id}")
def reset_session(session_id: str):
    """Clear memory for a given session."""
    cleared = clear_session(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {
        "status": "success",
        "message": f"Session memory cleared for '{session_id}'."
    }


@app.get("/api/documents")
def list_documents():
    """List all currently ingested documents in the knowledge base."""
    docs = get_ingested_documents()
    return {
        "count": len(docs),
        "documents": docs
    }


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """Delete an ingested source document from knowledge base and re-index vector store."""
    from app.rag import delete_document_and_reindex
    success, chunks_count, docs = delete_document_and_reindex(filename)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    return {
        "status": "success",
        "message": f"Successfully deleted document '{filename}'.",
        "total_chunks_indexed": chunks_count,
        "remaining_documents": docs
    }



@app.post("/api/ingest")
async def ingest_documents(files: list[UploadFile] = File(...)):
    """Upload multiple documents (.pdf, .md, .txt) and dynamically re-index the RAG vector store in batch."""
    import os
    from app.rag import reindex_all

    allowed_extensions = {".pdf", ".md", ".txt"}
    ingested_filenames = []

    for file in files:
        filename = file.filename or "uploaded_doc.txt"
        file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{file_ext}' for file '{filename}'. Allowed formats: {', '.join(allowed_extensions)}"
            )

        content_bytes = await file.read()
        if not content_bytes:
            continue

        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", filename)
        with open(file_path, "wb") as f:
            f.write(content_bytes)
        ingested_filenames.append(filename)

    if not ingested_filenames:
        raise HTTPException(status_code=400, detail="No valid, non-empty files were provided for ingestion.")

    chunks_count, all_docs = reindex_all()

    return {
        "status": "success",
        "message": f"Successfully ingested {len(ingested_filenames)} file(s) into vector storage.",
        "ingested_files": ingested_filenames,
        "total_chunks_indexed": chunks_count,
        "indexed_documents": all_docs
    }



# Mount Static Dashboard UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")


