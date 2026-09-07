# Autonomous Agent Architecture Overview

Our Research Agent uses a hybrid multi-modal architecture. 
It combines live real-time APIs (Open-Meteo Weather API and Wikipedia REST API) with a local vector database index powered by FAISS and SentenceTransformers.

Key Features:
1. Dynamic Tool Routing with Google Gemini 2.5 Flash.
2. Ingestion pipeline supporting PDF, Markdown, and TXT documents.
3. FastAPI REST server providing async prompt execution and document upload endpoints.
