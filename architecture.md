# Klaro Academic Chatbot - System Architecture

## Overview

Klaro is designed as a local-first, document-based AI chatbot that provides intelligent curriculum-based learning assistance. The system follows a Retrieval-Augmented Generation (RAG) architecture pattern, combining local document processing, vector search, and language model inference to deliver accurate, citation-backed responses.

## Core Architecture Components

### 1. Document Processing Pipeline
The document processing pipeline handles PDF ingestion, text extraction, and preparation for vector search:

- **PDF Loader**: Uses PyMuPDF (fitz) for robust PDF text extraction
- **Text Chunker**: Implements recursive character splitting with overlap for context preservation
- **Metadata Extractor**: Captures document source, page numbers, and section information
- **Preprocessing**: Cleans and normalizes text for optimal embedding generation

### 2. Vector Store and Retrieval System
The retrieval system provides fast, accurate document search capabilities:

- **Embedding Model**: Uses instructor-xl or bge-large-en for high-quality sentence embeddings
- **Vector Database**: FAISS for local, high-performance vector similarity search
- **Retrieval Strategy**: Hybrid approach combining semantic similarity and keyword matching
- **Context Assembly**: Intelligent context window management for LLM input

### 3. Language Model Integration
Local LLM integration ensures privacy and offline operation:

- **Model Runtime**: llama-cpp-python for efficient GGUF model inference
- **Supported Models**: Mistral-7B, Phi-3, OpenChat optimized for educational content
- **Prompt Engineering**: Specialized prompts for educational Q&A and summarization
- **Response Filtering**: Ensures responses are grounded in source material

### 4. Citation and Reference System
Comprehensive citation tracking maintains academic integrity:

- **Source Tracking**: Maintains document-to-chunk mappings with page/section metadata
- **Citation Generation**: Automatic hyperlink creation to source locations
- **Reference Validation**: Ensures all claims are backed by retrievable source material
- **Format Standardization**: Consistent citation format across all responses

## Data Flow Architecture

```
[PDF Documents] → [Text Extraction] → [Chunking] → [Embedding Generation] → [Vector Store]
                                                                                    ↓
[User Query] → [Query Embedding] → [Similarity Search] → [Context Assembly] → [LLM Inference] → [Response + Citations]
```

## Technical Stack Details

### Core Dependencies
- **PyMuPDF**: Fast, accurate PDF parsing with layout preservation
- **sentence-transformers**: State-of-the-art embedding models
- **langchain**: Modular RAG pipeline components
- **llama-cpp-python**: Optimized local LLM inference
- **faiss-cpu**: High-performance vector similarity search
- **gradio**: Simple, effective web UI framework

### Performance Considerations
- **Memory Management**: Efficient chunk loading and caching strategies
- **Inference Optimization**: Model quantization and context window optimization
- **Scalability**: Designed to handle 20+ PDFs with 500+ pages each
- **Response Time**: Target <3 seconds per query on standard hardware

## Security and Privacy Design

### Local-First Architecture
All processing occurs locally without external API calls:
- No data transmission to external services
- Complete offline operation capability
- User data remains on local machine
- No dependency on internet connectivity for core functionality

### Input Validation and Sanitization
- PDF content validation and malware scanning
- Query input sanitization to prevent injection attacks
- File type and size restrictions for document uploads
- Rate limiting to prevent resource exhaustion

## Database Schema (Phase 2 Implementation)

### Document Metadata Table
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    subject VARCHAR(100),
    grade_level VARCHAR(50),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size BIGINT,
    page_count INTEGER,
    checksum VARCHAR(64) UNIQUE
);
```

### Document Chunks Table
```sql
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section_title VARCHAR(500),
    content TEXT NOT NULL,
    embedding_vector VECTOR(768), -- Assuming 768-dim embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Query History Table
```sql
CREATE TABLE query_history (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT,
    cited_chunks INTEGER[],
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## API Design

### Core Endpoints

#### Document Management
- `POST /api/documents/upload` - Upload and process new documents
- `GET /api/documents` - List all processed documents
- `DELETE /api/documents/{id}` - Remove document and associated chunks
- `GET /api/documents/{id}/metadata` - Get document metadata and statistics

#### Query and Response
- `POST /api/query` - Submit question and receive AI response with citations
- `POST /api/summarize` - Generate topic summary across multiple documents
- `GET /api/search` - Perform semantic search across document corpus
- `GET /api/citations/{chunk_id}` - Retrieve full context for citation

#### System Management
- `GET /api/health` - System health and status check
- `GET /api/stats` - Usage statistics and performance metrics
- `POST /api/reindex` - Rebuild vector index for all documents

### Request/Response Formats

#### Query Request
```json
{
    "query": "What is photosynthesis?",
    "max_results": 5,
    "include_citations": true,
    "response_length": "medium"
}
```

#### Query Response
```json
{
    "response": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
    "citations": [
        {
            "document": "Grade8_Science.pdf",
            "page": 115,
            "section": "3.2 Plant Biology",
            "chunk_id": "doc_1_chunk_23",
            "relevance_score": 0.95
        }
    ],
    "response_time_ms": 1250,
    "confidence_score": 0.92
}
```

## Error Handling and Logging

### Error Categories
- **Document Processing Errors**: PDF corruption, unsupported formats, extraction failures
- **Model Inference Errors**: Memory limitations, model loading failures, timeout errors
- **Vector Search Errors**: Index corruption, embedding dimension mismatches
- **System Resource Errors**: Disk space, memory exhaustion, CPU overload

### Logging Strategy
- **Structured Logging**: JSON format for easy parsing and analysis
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL with appropriate filtering
- **Performance Metrics**: Query response times, model inference latency, memory usage
- **User Activity**: Query patterns, document usage, error frequencies

## Deployment Architecture

### Local Development
- Single-process Flask application with embedded vector store
- SQLite database for metadata storage
- Local file system for document storage
- Development UI via Gradio web interface

### Production Deployment
- Containerized deployment with Docker
- PostgreSQL database with pgvector extension
- Redis caching layer for frequently accessed chunks
- Load balancing for multiple inference workers
- Monitoring and alerting via Prometheus/Grafana

## Performance Optimization Strategies

### Model Optimization
- **Quantization**: Use 4-bit or 8-bit quantized models to reduce memory usage
- **Context Window Management**: Intelligent truncation and summarization of long contexts
- **Batch Processing**: Group similar queries for efficient inference
- **Model Caching**: Keep frequently used models in memory

### Vector Search Optimization
- **Index Tuning**: Optimize FAISS index parameters for speed vs accuracy trade-offs
- **Embedding Caching**: Cache embeddings for frequently queried content
- **Hierarchical Search**: Multi-stage retrieval for large document collections
- **Approximate Search**: Use approximate nearest neighbor for faster retrieval

### System Resource Management
- **Memory Pooling**: Efficient memory allocation and deallocation
- **Disk I/O Optimization**: Minimize file system operations through caching
- **CPU Utilization**: Multi-threading for parallel document processing
- **GPU Acceleration**: Optional CUDA support for faster inference

This architecture provides a solid foundation for building a robust, scalable, and maintainable academic chatbot system that meets all the requirements specified in the PRD while ensuring optimal performance and user experience.

