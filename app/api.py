from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.main import ask_agent
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


# Response schema
class ResearchResponse(BaseModel):
    query: str
    answer: str


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Research Agent API",
        "docs_url": "/docs"
    }


@app.post("/api/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    """Execute research prompt via AI Agent and return response."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        answer = ask_agent(request.prompt)
        return ResearchResponse(query=request.prompt, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

