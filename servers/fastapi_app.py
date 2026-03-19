"""
FastAPI Server for the MCP Assistant.
Provides API endpoints for image description, audio transcription, and document retrieval (RAG).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("./env.dev")

import traceback
import uuid
import base64
import torch
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM
from qwen_asr import Qwen3ASRModel
from qdrant_client import QdrantClient

from config import (
    FASTAPI_HOST, 
    FASTAPI_PORT,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    CHUNK_TOKENS,
    OVERLAP_TOKENS,
    DEFAULT_RETRIEVAL_LIMIT,
    DOCUMENT_STORAGE_DIR
)
from utils.rag_utils import (
    RAGPipeline,
    DocumentIngestion,
    DocumentChunker,
    ChunkIndexer,
    ChunkIngestion,
    RetrievalEngine,
    LLMContextValidator,
    ContextValidator
)


def load_asr_model():
    """Load the Qwen ASR model.
    
    Returns:
        Qwen3ASRModel instance
    """
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.float16,
        device_map="mps",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    return asr_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - loads and unloads ML models."""
    print(f"LOADING ASR MODEL...")
    app.state.asr_model = load_asr_model()
    print(f"LOADED ASR MODEL.")
    
    # Initialize RAG pipeline
    print(f"INITIALIZING RAG PIPELINE...")
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        app.state.rag_pipeline = RAGPipeline(
            qdrant_client=qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chunk_tokens=CHUNK_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
            storage_dir=DOCUMENT_STORAGE_DIR
        )
        print(f"RAG PIPELINE INITIALIZED.")
    except Exception as e:
        print(f"ERROR INITIALIZING RAG: {str(e)}")
        print(traceback.format_exc())
        app.state.rag_pipeline = None
    
    yield
    
    # Clean up the ML models and release the resources
    print(f"SHUTTING DOWN...")
    del app.state.asr_model
    del app.state.vl_chat_processor
    del app.state.vl_gpt
    del app.state.tokenizer
    if hasattr(app.state, 'rag_pipeline'):
        del app.state.rag_pipeline


app = FastAPI(title="API SERVER", lifespan=lifespan)


@app.get("/get-health")
def get_server_health():
    """Health check endpoint."""
    return {"status": "ok"}


# =============================================================================
# Audio Transcription
# =============================================================================

@app.get("/audio/transcribe")
async def transcribe_audio(request: Request, prompt: str, file_path: str) -> dict:
    """Transcribe an audio file using Qwen ASR model.
    
    Args:
        request: FastAPI request object
        prompt: Text prompt for transcription
        file_path: Path to the audio file
        
    Returns:
        Dictionary with status and transcription text
    """
    asr_model = request.app.state.asr_model
    try:
        result = asr_model.transcribe(
            audio=file_path,
            language=None,
        )
        
        return {
            "status": "success",
            "message": result[0].text,
        }
    except Exception as e:
        print(traceback.format_exc())
        print(f"[SERVER][TRANSCRIBE AUDIO] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error transcribing audio: {str(e)}"
        }


# =============================================================================
# RAG Document Retrieval Endpoints
# =============================================================================

@app.post("/document/ingest")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    chunk_tokens: int = Form(CHUNK_TOKENS),
    overlap_tokens: int = Form(OVERLAP_TOKENS)
) -> dict:
    """
    Ingest a document into the RAG system.
    
    This endpoint handles:
    1. Document ingestion: Download the uploaded document to a local file
    2. Chunk document: Chunk the document with adjustable parameters
    3. Chunk indexing: Create metadata for each chunk
    4. Chunk ingestion: Store chunks in Qdrant vector database
    
    Args:
        request: FastAPI request object
        file: Uploaded document file (PDF, DOCX, TXT)
        document_id: Optional custom document ID
        chunk_tokens: Number of tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        Dictionary with ingestion status and document info
    """
    try:
        rag_pipeline: RAGPipeline = request.app.state.rag_pipeline
        if not rag_pipeline:
            return {
                "status": "error",
                "message": "RAG pipeline not initialized"
            }
        
        rag_pipeline.document_chunker.number_of_tokens = chunk_tokens if chunk_tokens else CHUNK_TOKENS
        rag_pipeline.document_chunker.overlap_tokens = overlap_tokens if overlap_tokens else OVERLAP_TOKENS
        
        # Read file content
        content = await file.read()
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Ingest document
        result = rag_pipeline.ingest_document(
            content=content,
            filename=file.filename,
            document_id=document_id
        )
        
        return result
    
    except Exception as e:
        print(traceback.format_exc())
        print(f"[SERVER][INGEST DOCUMENT] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error ingesting document: {str(e)}"
        }


@app.get("/document/retrieval")
async def retrieve_documents(
    request: Request,
    query: str,
    document_id: Optional[str] = None,
    limit: int = 10,
    validate: bool = False
) -> dict:
    """
    Retrieve relevant document segments based on user query.
    
    This endpoint handles:
    1. The main retrieval function: Process user query to abstract intention
    2. Query the vector database to get retrieved vectors
    3. Vector to chunks: Trace vectors with appropriate chunks and metadata
    4. Context validation: Validate query with retrieved chunks (optional)
    5. Return the retrieved document segments through JSON response
    
    Args:
        request: FastAPI request object
        query: User query text
        document_id: Optional filter by specific document ID
        limit: Maximum number of results to return
        validate: Whether to run context validation
        
    Returns:
        Dictionary with retrieval results
    """
    try:
        rag_pipeline: RAGPipeline = request.app.state.rag_pipeline
        if not rag_pipeline:
            return {
                "status": "error",
                "message": "RAG pipeline not initialized"
            }
        
        # Retrieve documents
        result = await rag_pipeline.retrieve(
            query=query,
            limit=limit,
            document_id=document_id,
            validate=validate
        )
        
        return {
            "status": "success",
            "message": result,
        }
    
    except Exception as e:
        print(traceback.format_exc())
        print(f"[SERVER][RETRIEVE DOCUMENTS] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving documents: {str(e)}"
        }

@app.delete("/document/{document_id}")
async def delete_document(
    request: Request,
    document_id: str
) -> dict:
    """
    Delete a document and its associated chunks.
    
    Args:
        request: FastAPI request object
        document_id: ID of the document to delete
        
    Returns:
        Dictionary with deletion status
    """
    try:
        rag_pipeline: RAGPipeline = request.app.state.rag_pipeline
        if not rag_pipeline:
            return {
                "status": "error",
                "message": "RAG pipeline not initialized"
            }
        
        success = rag_pipeline.delete_document(document_id)
        
        return {
            "status": "success" if success else "error",
            "message": "Document deleted successfully" if success else "Failed to delete document"
        }
    
    except Exception as e:
        print(traceback.format_exc())
        print(f"[SERVER][DELETE DOCUMENT] Error: {str(e)}")
        return {
            "status": "error",
            "message": f"Error deleting document: {str(e)}"
        }


@app.get("/document/health")
async def get_rag_health(request: Request) -> dict:
    """
    Check RAG pipeline health status.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dictionary with RAG health status
    """
    try:
        rag_pipeline: RAGPipeline = request.app.state.rag_pipeline
        if not rag_pipeline:
            return {
                "status": "error",
                "message": "RAG pipeline not initialized",
                "healthy": False
            }
        
        return {
            "status": "success",
            "healthy": True,
            "collection": QDRANT_COLLECTION,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_tokens": CHUNK_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "healthy": False
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=FASTAPI_HOST, port=FASTAPI_PORT)

