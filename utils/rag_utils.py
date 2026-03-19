"""
RAG (Retrieval Augmented Generation) utilities for document processing and retrieval.
"""
import os
import uuid
import hashlib
import tempfile
import json
from pathlib import Path
from .logging_utils import logger
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# For document parsing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# For vector storage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Tokenizer for chunking
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# LLM/AI
from openai import AsyncOpenAI
from .constants import (
    INTENT_EXTRACTION_PROMP,
    DEFAULT_LLM_MODEL,
    VALIDATION_PROMPT,
    RAG_RESPONSE_PROMPT
)
from config import LLAMACPP_LLM_URL

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    chunk_id: str
    text: str
    document_id: str
    document_name: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with similarity score."""
    chunk: Chunk
    score: float
    reason: Optional[str] = None


# =============================================================================
# Document Ingestion
# =============================================================================

class DocumentIngestion:
    """Handles document download/upload to local storage."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize document ingestion.
        
        Args:
            storage_dir: Directory to store uploaded documents. 
                        Defaults to temp directory.
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_dir = Path(tempfile.gettempdir()) / "rag_documents"
            self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_document(self, content: bytes, filename: str) -> str:
        """
        Save uploaded document to local storage.
        
        Args:
            content: Raw file content bytes
            filename: Original filename
            
        Returns:
            Path to saved document
        """
        # Generate unique filename to avoid collisions
        file_hash = hashlib.md5(content).hexdigest()[:8]
        unique_filename = f"{file_hash}_{filename}"
        file_path = self.storage_dir / unique_filename
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return str(file_path)
    
    def save_from_base64(self, base64_content: str, filename: str) -> str:
        """
        Save base64 encoded document to local storage.
        
        Args:
            base64_content: Base64 encoded file content
            filename: Original filename
            
        Returns:
            Path to saved document
        """
        import base64
        content = base64.b64decode(base64_content)
        return self.save_document(content, filename)
    
    def delete_document(self, file_path: str) -> bool:
        """
        Delete a document from storage.
        
        Args:
            file_path: Path to document
            
        Returns:
            True if deleted successfully
        """
        try:
            Path(file_path).unlink(missing_ok=True)
            return True
        except Exception:
            return False


# =============================================================================
# Document Chunking
# =============================================================================

class DocumentChunker:
    """Chunks documents into smaller segments with adjustable parameters."""
    
    def __init__(
        self, 
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        number_of_tokens: int = 512,
        overlap_tokens: int = 50
    ):
        """
        Initialize document chunker.
        
        Args:
            tokenizer_name: Name of tokenizer model for token counting
            number_of_tokens: Target number of tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
        """
        self.number_of_tokens = number_of_tokens
        self.overlap_tokens = overlap_tokens
        
        if AutoTokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, device="mps")
            except Exception:
                # Fallback to simple word-based chunking
                self.tokenizer = None
        else:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Simple word-based approximation
            return len(text.split())
    
    def chunk_text(self, text: str, document_id: str, document_name: str) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            document_id: Unique identifier for the document
            document_name: Name of the document
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        if self.tokenizer:
            # Token-based chunking
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunk_size = self.number_of_tokens
            overlap = self.overlap_tokens
            
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                
                if chunk_text.strip():
                    chunk = Chunk(
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        text=chunk_text,
                        document_id=document_id,
                        document_name=document_name,
                        start_index=i,
                        end_index=i + len(chunk_tokens),
                        metadata={
                            "token_count": len(chunk_tokens),
                            "chunk_index": len(chunks)
                        }
                    )
                    chunks.append(chunk)
                
                if i + chunk_size >= len(tokens):
                    break
        else:
            # Simple word-based chunking (fallback)
            words = text.split()
            chunk_size = self.number_of_tokens * 4  # Approximate: 1 token ≈ 4 chars
            overlap = self.overlap_tokens * 4
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if chunk_text.strip():
                    chunk = Chunk(
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        text=chunk_text,
                        document_id=document_id,
                        document_name=document_name,
                        start_index=i,
                        end_index=i + len(chunk_words),
                        metadata={
                            "word_count": len(chunk_words),
                            "chunk_index": len(chunks)
                        }
                    )
                    chunks.append(chunk)
                
                if i + chunk_size >= len(words):
                    break
        
        return chunks
    
    def chunk_file(self, file_path: str, document_id: str) -> List[Chunk]:
        """
        Chunk a document file.
        
        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            
        Returns:
            List of Chunk objects
        """
        file_path = Path(file_path)
        document_name = file_path.name
        extension = file_path.suffix.lower()
        
        # Extract text based on file type
        if extension == '.pdf':
            text = self._extract_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            text = self._extract_docx(file_path)
        elif extension == '.txt':
            text = self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return self.chunk_text(text, document_id, document_name)
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        if not PyPDF2:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        if not docx:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
        
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_txt(self, file_path: Path) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()


# =============================================================================
# Chunk Indexing
# =============================================================================

class ChunkIndexer:
    """Creates metadata index for chunks."""
    
    def __init__(self):
        """Initialize chunk indexer."""
        self.index: Dict[str, Chunk] = {}
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index."""
        for chunk in chunks:
            self.index[chunk.chunk_id] = chunk
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a chunk by ID."""
        return self.index.get(chunk_id)
    
    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Get all chunks for a specific document."""
        return [
            chunk for chunk in self.index.values() 
            if chunk.document_id == document_id
        ]
    
    def remove_document_chunks(self, document_id: str) -> None:
        """Remove all chunks for a specific document."""
        self.index = {
            cid: chunk for cid, chunk in self.index.items()
            if chunk.document_id != document_id
        }
    
    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all indexed chunks."""
        return [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "metadata": chunk.metadata
            }
            for chunk in self.index.values()
        ]


# =============================================================================
# Chunk Ingestion (Qdrant)
# =============================================================================

class ChunkIngestion:
    """Handles chunk ingestion to Qdrant vector database."""
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "document_chunks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_size: int = 384
    ):
        """
        Initialize chunk ingestion.
        
        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of the collection to use
            embedding_model: Name of the sentence transformer model
            vector_size: Dimension of embedding vectors
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize embedding model
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
            except Exception:
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # Create collection if not exists
        self._ensure_collection()
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        if self.embedding_model:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # Return random vectors as placeholder if no model
            import numpy as np
            return np.random.rand(len(texts), self.vector_size).tolist()
    
    def ingest_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Ingest chunks into Qdrant.
        
        Args:
            chunks: List of Chunk objects to ingest
            
        Returns:
            True if successful
        """
        if not chunks:
            return True
        
        # Get embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)
        
        # Create points for insertion
        points = []
        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=uuid.uuid4(),
                vector=embeddings[i],
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document_name,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "metadata": json.dumps(chunk.metadata)
                }
            )
            points.append(point)
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        return True
    
    def delete_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete chunks for
            
        if successful
        Returns:
            True """
        # Get all chunk IDs for this document
        filter_condition = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )
        
        # Delete points
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_condition,
            wait=True
        )
        
        return True
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            limit: Maximum number of results
            document_id: Optional filter by document ID
            
        Returns:
            List of RetrievalResult objects
        """
        # Get query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Build filter if document_id provided
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
        
        # Search Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        
        # Convert to RetrievalResult
        retrieval_results = []
        for point in results.points:
            chunk = Chunk(
                chunk_id=point.payload["chunk_id"],
                text=point.payload["text"],
                document_id=point.payload["document_id"],
                document_name=point.payload["document_name"],
                start_index=point.payload["start_index"],
                end_index=point.payload["end_index"],
                metadata=json.loads(point.payload.get("metadata", "{}"))
            )
            retrieval_results.append(RetrievalResult(
                chunk=chunk,
                score=point.score
            ))
        
        return retrieval_results


# =============================================================================
# Context Validator (Boilerplate - for user implementation)
# =============================================================================

class ContextValidator(ABC):
    """
    Abstract base class for context validation.
    This is a boilerplate class - implement your own validation logic.
    """
    
    @abstractmethod
    def validate(
        self, 
        query: str, 
        retrieved_chunks: List[RetrievalResult]
    ) -> Tuple[bool, str]:
        """
        Validate if retrieved chunks are relevant to the query.
        
        Args:
            query: Original user query
            retrieved_chunks: List of retrieved chunks with scores
            
        Returns:
            Tuple of (is_valid, reason)
        """
        pass
    
    @abstractmethod
    def filter_chunks(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Filter and rank retrieved chunks based on relevance.
        
        Args:
            query: Original user query
            retrieved_chunks: List of retrieved chunks with scores
            
        Returns:
            Filtered list of RetrievalResult objects
        """
        pass


class LLMContextValidator(ContextValidator):
    """
    LLM-based context validator (boilerplate).
    Implement the actual LLM call in the methods below.
    """
    
    def __init__(self):
        """
        Initialize LLM context validator.
        
        Args:
            llm_client: Optional LLM client for making inference calls
        """
        self.slm_validator = AsyncOpenAI(
            base_url=f"{LLAMACPP_LLM_URL}/v1",
            api_key='llama.cpp',  # required, but unused
        )
        self.validator_model_name = DEFAULT_LLM_MODEL # DEFAULT_SLM_MODEL
    
    async def extract_intention(self, query: str):
        self.messages = [
            {
                "role": "system",
                "content": INTENT_EXTRACTION_PROMP.format(query_message=query)
            }
        ]
        self.messages.append({"role": "user", "content": f"Please give me the intention response."})
        
        response = await self.slm_validator.chat.completions.create(
            model=self.validator_model_name,
            messages=self.messages
        )
        logger.info(f"[RAG][INTENTION]: {response.choices[0].message.content}")
        
        return response.choices[0].message.content
    
    async def validate(
        self, 
        query: str, 
        retrieved_chunks: List[RetrievalResult]
    ) -> Tuple[bool, str]:
        """
        Validate if retrieved chunks are relevant to the query.
        
        Args:
            query: Original user query
            retrieved_chunks: List of retrieved chunks with scores
            
        Returns:
            Tuple of (is_valid, reason)
        """
        self.messages = [
            {
                "role": "system",
                "content": VALIDATION_PROMPT.format(
                    intention=query,
                    retrieved_document=retrieved_chunks
                )
            }
        ]
        self.messages.append({"role": "user", "content": f"Please give me the intention response."})
        
        response = await self.slm_validator.chat.completions.create(
            model=self.validator_model_name,
            messages=self.messages
        )
        logger.info(f"[RAG][VALIDATION]: {response.choices[0].message.content}")
        
        # Default: assume valid if there are chunks
        if not retrieved_chunks:
            return False, "No chunks retrieved"
        
        # Check minimum score threshold
        avg_score = sum(r.score for r in retrieved_chunks) / len(retrieved_chunks)
        if avg_score < 0.3:
            return False, f"Low similarity scores (avg: {avg_score:.3f})"
        
        return True, response.choices[0].message.content
    
    def filter_chunks(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Filter and rank retrieved chunks based on relevance.
        
        Args:
            query: Original user query
            retrieved_chunks: List of retrieved chunks with scores
            
        Returns:
            Filtered list of RetrievalResult objects
        """
        # Example: Use LLM to score each chunk's relevance
        
        # Default: return all chunks above threshold
        threshold = 0.3
        filtered = [r for r in retrieved_chunks if r.score >= threshold]
        
        # Sort by score
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        return filtered

# =============================================================================
# Retrieval Engine
# =============================================================================

class RetrievalEngine:
    """Main retrieval engine that processes queries and retrieves chunks."""
    
    def __init__(
        self,
        chunk_ingestion: ChunkIngestion,
        intent_processor: LLMContextValidator | None = None
    ):
        """
        Initialize retrieval engine.
        
        Args:
            chunk_ingestion: ChunkIngestion instance for vector storage
            intent_processor: Optional processor for converting query to intent
        """
        self.chunk_ingestion = chunk_ingestion
        self.intent_processor = intent_processor
        self.response_prompt = RAG_RESPONSE_PROMPT
    
    async def get_response_prompt(self, retrieved_document: list):
        return {
            "role": "assistant", 
            "content": self.response_prompt.format(
                retrieved_document=[i["text"] for i in retrieved_document]
            )
        }
    async def process_query(self, user_query: str) -> str:
        """
        Process user query to abstract intention for vector query.
        
        Args:
            user_query: Raw user query
            
        Returns:
            Processed intent/query for vector search
        """
        # If intent processor is provided, use it
        if self.intent_processor:
            return await self.intent_processor.extract_intention(user_query)
        
        # Otherwise, use the query as-is
        return user_query
    
    async def retrieve(
        self,
        user_query: str,
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> Tuple[List[RetrievalResult], str]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            user_query: User's query text
            limit: Maximum number of results
            document_id: Optional filter by document ID
            
        Returns:
            List of RetrievalResult objects
        """
        # Process query to get intent
        intent = await self.process_query(user_query)
        
        # Search vector database
        results = self.chunk_ingestion.search(
            query=intent,
            limit=limit,
            document_id=document_id
        )
        
        return results, intent
    
    async def retrieve_with_metadata(
        self,
        user_query: str,
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve chunks with full metadata.
        
        Args:
            user_query: User's query text
            limit: Maximum number of results
            document_id: Optional filter by document ID
            
        Returns:
            Dictionary with retrieval results and metadata
        """
        results = await self.retrieve(user_query, limit, document_id)
        
        return {
            "status": "success",
            "query": user_query,
            "results": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "text": r.chunk.text,
                    "document_id": r.chunk.document_id,
                    "document_name": r.chunk.document_name,
                    "score": r.score,
                    "metadata": r.chunk.metadata
                }
                for r in results
            ]
        }


# =============================================================================
# RAG Pipeline (Convenience Class)
# =============================================================================

class RAGPipeline:
    """Complete RAG pipeline combining all components."""
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "document_chunks",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_tokens: int = 512,
        overlap_tokens: int = 50,
        storage_dir: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of collection to use
            embedding_model: Sentence transformer model name
            chunk_tokens: Number of tokens per chunk
            overlap_tokens: Overlap between chunks
            storage_dir: Directory for document storage
            validator: Optional context validator
        """
        # Initialize components
        self.document_ingestion = DocumentIngestion(storage_dir)
        self.document_chunker = DocumentChunker(
            number_of_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens
        )
        self.chunk_indexer = ChunkIndexer()
        self.chunk_ingestion = ChunkIngestion(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        self.retrieval_engine = RetrievalEngine(
            chunk_ingestion=self.chunk_ingestion,
            intent_processor=LLMContextValidator()
        )
        self.validator = LLMContextValidator()
    
    def ingest_document(
        self,
        content: bytes,
        filename: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            content: Document content bytes
            filename: Original filename
            document_id: Optional custom document ID
            
        Returns:
            Status dictionary
        """
        import uuid
        
        if not document_id:
            document_id = str(uuid.uuid4())
        
        # Save document
        file_path = self.document_ingestion.save_document(content, filename)
        
        # Chunk document
        chunks = self.document_chunker.chunk_file(file_path, document_id)
        # print(f"[RAG] - CHUNKS: {chunks}")
        
        # Index chunks
        self.chunk_indexer.add_chunks(chunks)
        
        # Ingest to vector database
        self.chunk_ingestion.ingest_chunks(chunks)
        
        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "num_chunks": len(chunks),
            "file_path": file_path
        }
    
    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant document segments.
        
        Args:
            query: User query
            limit: Maximum number of results
            document_id: Optional filter by document
            validate: Whether to run context validation
            
        Returns:
            Retrieval results dictionary
        """
        # Retrieve chunks
        results, intent = await self.retrieval_engine.retrieve(query, limit, document_id)
        
        if not results:
            return {
                "status": "success",
                "query": intent,
                "validated": False,
                "results": [],
                "message": "No relevant chunks found"
            }
        
        # Validate if requested
        is_valid = validate
        validation_reason = ""
        if validate:
            is_valid, validation_reason = await self.validator.validate(query, results)
        
        # Filter chunks if validation passed
        # if is_valid:
        #     filtered_results = self.validator.filter_chunks(query, results)
        # else:
        #     filtered_results = []
        filtered_results = results
        
        return {
            "status": "success",
            "query": intent,
            "validated": validate,
            "validation_passed": is_valid,
            "validation_reason": validation_reason,
            "results": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "text": r.chunk.text,
                    "document_id": r.chunk.document_id,
                    "document_name": r.chunk.document_name,
                    "score": r.score,
                    "metadata": r.chunk.metadata
                }
                for r in filtered_results
            ]
        }
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its chunks.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        self.chunk_indexer.remove_document_chunks(document_id)
        return self.chunk_ingestion.delete_chunks(document_id)

