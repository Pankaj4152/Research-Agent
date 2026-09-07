import uuid
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.main import ask_agent, clear_session, get_session_history
from app.rag import ingest_file_and_reindex

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


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Research Agent API",
        "docs_url": "/docs"
    }


@app.post("/api/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    """Execute research prompt via AI Agent and return response with multi-turn session state."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        answer, session_id = ask_agent(request.prompt, session_id=request.session_id)
        return ResearchResponse(query=request.prompt, answer=answer, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload a document (.pdf, .md, .txt) and dynamically re-index the RAG vector store."""
    allowed_extensions = {".pdf", ".md", ".txt"}
    filename = file.filename or "uploaded_doc.txt"
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(allowed_extensions)}"
        )

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        chunks_count, all_docs = ingest_file_and_reindex(filename, content_bytes)
        return {
            "status": "success",
            "message": f"Successfully ingested '{filename}' into vector storage.",
            "filename": filename,
            "total_chunks_indexed": chunks_count,
            "indexed_documents": all_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


# Mount Static Dashboard UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")


