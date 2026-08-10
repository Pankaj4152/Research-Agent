from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.main import ask_agent

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
