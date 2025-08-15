"""
Vector Store Module for Klaro Academic Chatbot

This module handles vector embeddings generation, storage, and similarity search
using FAISS for high-performance local vector operations. It provides the core
retrieval functionality for the RAG pipeline.
"""

import numpy as np
import pickle
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
from text_processor import ProcessedChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Structure for search results with relevance scoring"""
    chunk_id: str
    content: str
    document_name: str
    page_number: int
    section_title: Optional[str]
    similarity_score: float
    rank: int

class VectorStore:
    """
    Handles vector embeddings and similarity search for document chunks.
    
    Features:
    - High-quality sentence embeddings using transformer models
    - Fast similarity search with FAISS indexing
    - Persistent storage and loading of embeddings
    - Hybrid search combining semantic and keyword matching
    - Relevance scoring and result ranking
    """
    
    def __init__(self, 
                 model_name: str = "hkunlp/instructor-xl",
                 embeddings_dir: str = "./embeddings/",
                 index_name: str = "klaro_index"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model
            embeddings_dir: Directory to store embedding files
            index_name: Name for the FAISS index files
        """
        self.model_name = model_name
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.index_name = index_name
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.chunk_id_to_index: Dict[str, int] = {}
        
        # Load existing index if available
        self._load_index()
    
    def add_chunks(self, chunks: List[ProcessedChunk]) -> None:
        """
        Add processed chunks to the vector store.
        
        Args:
            chunks: List of processed chunks to add
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts, 
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Initialize or update FAISS index
        if self.index is None:
            self._initialize_index(embeddings.shape[1])
        
        ### Add embeddings to index
        # start_index = self.index.ntotal
        # self.index.add(embeddings)
        
        # Add embeddings with IDs
        start_index = self.index.ntotal
        ids = np.arange(start_index, start_index + len(embeddings))
        self.index.add_with_ids(embeddings, ids)

        
        # Update metadata
        for i, chunk in enumerate(chunks):
            chunk_index = start_index + i
            self.chunk_id_to_index[chunk.chunk_id] = chunk_index
            
            metadata = {
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "content": chunk.content,
                "word_count": chunk.word_count,
                "chunk_index": chunk.chunk_index
            }
            self.chunk_metadata.append(metadata)
        
        logger.info(f"Successfully added {len(chunks)} chunks. Total chunks: {self.index.ntotal}")
        
        # Save the updated index
        self._save_index()
    
    def _initialize_index(self, dimension: int) -> None:
        """
        Initialize a new FAISS index.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        logger.info(f"Initializing FAISS index with dimension {dimension}")
        
        # Use IndexFlatIP for exact cosine similarity search
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize vectors for cosine similarity
        self.index = faiss.IndexIDMap(self.index)
    
    def search(self, 
               query: str, 
               k: int = 5,
               min_similarity: float = 0.0) -> List[SearchResult]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results ranked by similarity
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No embeddings in vector store")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        similarities, indices = self.index.search(query_embedding, k)
        
        # Convert results to SearchResult objects
        results = []
        for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1 or similarity < min_similarity:
                continue
            
            if idx >= len(self.chunk_metadata):
                logger.warning(f"Index {idx} out of range for metadata")
                continue
            
            metadata = self.chunk_metadata[idx]
            
            result = SearchResult(
                chunk_id=metadata["chunk_id"],
                content=metadata["content"],
                document_name=metadata["document_name"],
                page_number=metadata["page_number"],
                section_title=metadata["section_title"],
                similarity_score=float(similarity),
                rank=rank + 1
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} results for query: '{query[:50]}...'")
        return results
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query text
            k: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            
        Returns:
            List of search results with combined scoring
        """
        # Get semantic search results
        semantic_results = self.search(query, k * 2)  # Get more for reranking
        
        # Perform keyword matching
        keyword_scores = self._keyword_search(query)
        
        # Combine scores
        combined_results = []
        for result in semantic_results:
            keyword_score = keyword_scores.get(result.chunk_id, 0.0)
            
            # Combine semantic and keyword scores
            combined_score = (
                semantic_weight * result.similarity_score +
                keyword_weight * keyword_score
            )
            
            # Update the result with combined score
            result.similarity_score = combined_score
            combined_results.append(result)
        
        # Sort by combined score and return top k
        combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results[:k]):
            result.rank = i + 1
        
        return combined_results[:k]
    
    def _keyword_search(self, query: str) -> Dict[str, float]:
        """
        Perform keyword-based search scoring.
        
        Args:
            query: Search query text
            
        Returns:
            Dictionary mapping chunk_id to keyword score
        """
        query_words = set(query.lower().split())
        scores = {}
        
        for metadata in self.chunk_metadata:
            content_words = set(metadata["content"].lower().split())
            
            # Calculate Jaccard similarity
            intersection = query_words.intersection(content_words)
            union = query_words.union(content_words)
            
            if union:
                jaccard_score = len(intersection) / len(union)
                scores[metadata["chunk_id"]] = jaccard_score
        
        return scores
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            Chunk metadata if found, None otherwise
        """
        for metadata in self.chunk_metadata:
            if metadata["chunk_id"] == chunk_id:
                return metadata
        return None
    
    def get_chunks_by_document(self, document_name: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of chunk metadata for the document
        """
        return [metadata for metadata in self.chunk_metadata 
                if metadata["document_name"] == document_name]
    
    def remove_document(self, document_name: str) -> int:
        """
        Remove all chunks for a specific document.
        
        Args:
            document_name: Name of the document to remove
            
        Returns:
            Number of chunks removed
        """
        # Find chunks to remove
        chunks_to_remove = [
            i for i, metadata in enumerate(self.chunk_metadata)
            if metadata["document_name"] == document_name
        ]
        
        if not chunks_to_remove:
            logger.info(f"No chunks found for document: {document_name}")
            return 0
        
        # Note: FAISS doesn't support efficient removal of specific vectors
        # For production use, consider rebuilding the index or using a different approach
        logger.warning("Document removal requires rebuilding the index")
        
        # Remove from metadata
        for i in reversed(chunks_to_remove):
            removed_metadata = self.chunk_metadata.pop(i)
            chunk_id = removed_metadata["chunk_id"]
            if chunk_id in self.chunk_id_to_index:
                del self.chunk_id_to_index[chunk_id]
        
        logger.info(f"Removed {len(chunks_to_remove)} chunks for document: {document_name}")
        return len(chunks_to_remove)
    
    def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            index_path = self.embeddings_dir / f"{self.index_name}.faiss"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.embeddings_dir / f"{self.index_name}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    "chunk_metadata": self.chunk_metadata,
                    "chunk_id_to_index": self.chunk_id_to_index,
                    "model_name": self.model_name,
                    "embedding_dimension": self.embedding_dimension
                }, f)
            
            logger.info(f"Saved index and metadata to {self.embeddings_dir}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def _load_index(self) -> bool:
        """
        Load existing FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = self.embeddings_dir / f"{self.index_name}.faiss"
            metadata_path = self.embeddings_dir / f"{self.index_name}_metadata.pkl"
            
            if not (index_path.exists() and metadata_path.exists()):
                logger.info("No existing index found")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.chunk_metadata = data["chunk_metadata"]
                self.chunk_id_to_index = data["chunk_id_to_index"]
                
                # Verify model compatibility
                if data["model_name"] != self.model_name:
                    logger.warning(f"Model mismatch: {data['model_name']} vs {self.model_name}")
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def rebuild_index(self, chunks: List[ProcessedChunk]) -> None:
        """
        Rebuild the entire index from scratch.
        
        Args:
            chunks: All chunks to include in the new index
        """
        logger.info("Rebuilding vector store index")
        
        # Clear existing data
        self.index = None
        self.chunk_metadata = []
        self.chunk_id_to_index = {}
        
        # Add all chunks
        self.add_chunks(chunks)
        
        logger.info("Index rebuild complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        if self.index is None:
            return {
                "total_vectors": 0,
                "embedding_dimension": self.embedding_dimension,
                "model_name": self.model_name,
                "index_size_mb": 0
            }
        
        # Calculate approximate index size
        index_size_bytes = self.index.ntotal * self.embedding_dimension * 4  # float32
        index_size_mb = round(index_size_bytes / (1024 * 1024), 2)
        
        # Document statistics
        documents = set(metadata["document_name"] for metadata in self.chunk_metadata)
        
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "index_size_mb": index_size_mb,
            "total_documents": len(documents),
            "chunks_per_document": round(self.index.ntotal / len(documents), 1) if documents else 0
        }


# Example usage and testing
if __name__ == "__main__":
    from text_processor import TextProcessor, ProcessedChunk
    
    # Sample chunks for testing
    sample_chunks = [
        ProcessedChunk(
            chunk_id="test_doc_chunk_0001",
            document_name="test_biology.pdf",
            page_number=1,
            section_title="Introduction to Biology",
            content="Biology is the scientific study of life and living organisms. It encompasses molecular biology, genetics, ecology, and evolution.",
            char_start=0,
            char_end=120,
            word_count=20,
            chunk_index=0,
            overlap_with_previous=False,
            overlap_with_next=True
        ),
        ProcessedChunk(
            chunk_id="test_doc_chunk_0002",
            document_name="test_biology.pdf",
            page_number=1,
            section_title="Cell Structure",
            content="Cells are the basic units of life. They contain organelles such as the nucleus, mitochondria, and ribosomes that perform specific functions.",
            char_start=100,
            char_end=220,
            word_count=22,
            chunk_index=1,
            overlap_with_previous=True,
            overlap_with_next=False
        )
    ]
    
    # Initialize vector store
    vector_store = VectorStore(
        model_name="all-MiniLM-L6-v2",  # Smaller model for testing
        embeddings_dir="./test_embeddings/"
    )
    
    # Add chunks
    vector_store.add_chunks(sample_chunks)
    
    # Test search
    results = vector_store.search("What is biology?", k=2)
    
    print(f"Search Results:")
    for result in results:
        print(f"- Rank {result.rank}: {result.chunk_id}")
        print(f"  Similarity: {result.similarity_score:.3f}")
        print(f"  Content: {result.content[:100]}...")
        print()
    
    # Print statistics
    stats = vector_store.get_stats()
    print(f"Vector Store Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

