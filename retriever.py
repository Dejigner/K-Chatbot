"""
Retriever Module for Klaro Academic Chatbot

This module handles document retrieval, context assembly, and citation generation
for the RAG pipeline. It provides the interface between user queries and the
vector store, with intelligent context management and source tracking.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from vector_store import VectorStore, SearchResult
from text_processor import ProcessedChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Structure for retrieval results with context and citations"""
    query: str
    context: str
    citations: List[Dict[str, Any]]
    total_chunks: int
    retrieval_score: float
    processing_time_ms: int

@dataclass
class Citation:
    """Structure for citation information"""
    document_name: str
    page_number: int
    section_title: Optional[str]
    chunk_id: str
    relevance_score: float
    excerpt: str

class DocumentRetriever:
    """
    Handles document retrieval and context assembly for RAG queries.
    
    Features:
    - Intelligent context assembly from multiple sources
    - Citation generation with source tracking
    - Query expansion and refinement
    - Context window management for optimal LLM input
    - Relevance scoring and filtering
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 max_context_length: int = 4000,
                 max_chunks_per_query: int = 5,
                 min_relevance_score: float = 0.1):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: Vector store instance for similarity search
            max_context_length: Maximum length of assembled context
            max_chunks_per_query: Maximum number of chunks to retrieve
            min_relevance_score: Minimum relevance threshold
        """
        self.vector_store = vector_store
        self.max_context_length = max_context_length
        self.max_chunks_per_query = max_chunks_per_query
        self.min_relevance_score = min_relevance_score
        
        # Query processing statistics
        self.query_count = 0
        self.total_processing_time = 0
    
    def retrieve_context(self, 
                        query: str, 
                        use_hybrid_search: bool = True) -> RetrievalResult:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query text
            use_hybrid_search: Whether to use hybrid semantic+keyword search
            
        Returns:
            RetrievalResult with context and citations
        """
        import time
        start_time = time.time()
        
        logger.info(f"Retrieving context for query: '{query[:50]}...'")
        
        # Perform search
        if use_hybrid_search:
            search_results = self.vector_store.hybrid_search(
                query, 
                k=self.max_chunks_per_query,
                semantic_weight=0.7,
                keyword_weight=0.3
            )
        else:
            search_results = self.vector_store.search(
                query, 
                k=self.max_chunks_per_query,
                min_similarity=self.min_relevance_score
            )
        
        # Filter results by relevance
        filtered_results = [
            result for result in search_results 
            if result.similarity_score >= self.min_relevance_score
        ]
        
        if not filtered_results:
            logger.warning(f"No relevant results found for query: '{query}'")
            return RetrievalResult(
                query=query,
                context="",
                citations=[],
                total_chunks=0,
                retrieval_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Assemble context and citations
        context, citations = self._assemble_context_and_citations(filtered_results)
        
        # Calculate overall retrieval score
        retrieval_score = self._calculate_retrieval_score(filtered_results)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Update statistics
        self.query_count += 1
        self.total_processing_time += processing_time
        
        result = RetrievalResult(
            query=query,
            context=context,
            citations=citations,
            total_chunks=len(filtered_results),
            retrieval_score=retrieval_score,
            processing_time_ms=processing_time
        )
        
        logger.info(f"Retrieved {len(filtered_results)} chunks in {processing_time}ms")
        return result
    
    def _assemble_context_and_citations(self, 
                                       search_results: List[SearchResult]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assemble context text and generate citations from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            Tuple of (assembled_context, citations_list)
        """
        context_parts = []
        citations = []
        current_length = 0
        
        # Group results by document for better organization
        results_by_doc = {}
        for result in search_results:
            doc_name = result.document_name
            if doc_name not in results_by_doc:
                results_by_doc[doc_name] = []
            results_by_doc[doc_name].append(result)
        
        # Process results document by document
        for doc_name, doc_results in results_by_doc.items():
            # Sort by page number and relevance
            doc_results.sort(key=lambda x: (x.page_number, -x.similarity_score))
            
            for result in doc_results:
                # Check if adding this chunk would exceed context length
                chunk_text = result.content
                if current_length + len(chunk_text) > self.max_context_length:
                    # Try to fit a truncated version
                    remaining_space = self.max_context_length - current_length - 100
                    if remaining_space > 200:  # Only if we have reasonable space
                        chunk_text = chunk_text[:remaining_space] + "..."
                    else:
                        break  # Skip this and remaining chunks
                
                # Add context with source attribution
                context_part = f"[Source: {doc_name}, Page {result.page_number}]\n{chunk_text}\n"
                context_parts.append(context_part)
                current_length += len(context_part)
                
                # Create citation
                citation = {
                    "document_name": doc_name,
                    "page_number": result.page_number,
                    "section_title": result.section_title,
                    "chunk_id": result.chunk_id,
                    "relevance_score": result.similarity_score,
                    "excerpt": self._create_excerpt(chunk_text, 150)
                }
                citations.append(citation)
        
        # Assemble final context
        context = "\n---\n".join(context_parts)
        
        return context, citations
    
    def _create_excerpt(self, text: str, max_length: int = 150) -> str:
        """
        Create a brief excerpt from text for citation purposes.
        
        Args:
            text: Full text content
            max_length: Maximum length of excerpt
            
        Returns:
            Text excerpt
        """
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundary
        truncated = text[:max_length]
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence_end > max_length * 0.7:  # If we found a good break point
            return truncated[:last_sentence_end + 1]
        else:
            # Break at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.7:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."
    
    def _calculate_retrieval_score(self, search_results: List[SearchResult]) -> float:
        """
        Calculate overall retrieval quality score.
        
        Args:
            search_results: List of search results
            
        Returns:
            Retrieval quality score (0.0 to 1.0)
        """
        if not search_results:
            return 0.0
        
        # Average similarity score weighted by rank
        weighted_scores = []
        for i, result in enumerate(search_results):
            # Higher weight for top results
            weight = 1.0 / (i + 1)
            weighted_scores.append(result.similarity_score * weight)
        
        # Normalize by total weight
        total_weight = sum(1.0 / (i + 1) for i in range(len(search_results)))
        
        return sum(weighted_scores) / total_weight
    
    def retrieve_for_summarization(self, 
                                  topic: str, 
                                  max_chunks: int = 10) -> RetrievalResult:
        """
        Retrieve context specifically for topic summarization.
        
        Args:
            topic: Topic to summarize
            max_chunks: Maximum number of chunks to retrieve
            
        Returns:
            RetrievalResult optimized for summarization
        """
        logger.info(f"Retrieving context for summarization: '{topic}'")
        
        # Use broader search for summarization
        search_results = self.vector_store.hybrid_search(
            topic,
            k=max_chunks,
            semantic_weight=0.8,  # Emphasize semantic similarity for topics
            keyword_weight=0.2
        )
        
        # Filter and diversify results for better coverage
        diversified_results = self._diversify_results(search_results, topic)
        
        # Assemble context with document diversity in mind
        context, citations = self._assemble_summarization_context(diversified_results)
        
        return RetrievalResult(
            query=topic,
            context=context,
            citations=citations,
            total_chunks=len(diversified_results),
            retrieval_score=self._calculate_retrieval_score(diversified_results),
            processing_time_ms=0  # Not tracking for summarization
        )
    
    def _diversify_results(self, 
                          search_results: List[SearchResult], 
                          topic: str) -> List[SearchResult]:
        """
        Diversify search results to cover different aspects of a topic.
        
        Args:
            search_results: Original search results
            topic: Topic being searched
            
        Returns:
            Diversified list of search results
        """
        if len(search_results) <= 5:
            return search_results
        
        # Group by document to ensure diversity
        doc_groups = {}
        for result in search_results:
            doc_name = result.document_name
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(result)
        
        # Select top results from each document
        diversified = []
        max_per_doc = max(2, len(search_results) // len(doc_groups))
        
        for doc_name, doc_results in doc_groups.items():
            # Sort by relevance and take top results
            doc_results.sort(key=lambda x: x.similarity_score, reverse=True)
            diversified.extend(doc_results[:max_per_doc])
        
        # Sort final results by relevance
        diversified.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return diversified
    
    def _assemble_summarization_context(self, 
                                       search_results: List[SearchResult]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assemble context specifically for summarization tasks.
        
        Args:
            search_results: Search results to assemble
            
        Returns:
            Tuple of (context, citations) optimized for summarization
        """
        # Group by document for organized presentation
        doc_sections = {}
        citations = []
        
        for result in search_results:
            doc_name = result.document_name
            if doc_name not in doc_sections:
                doc_sections[doc_name] = []
            
            doc_sections[doc_name].append(result)
            
            # Create citation
            citation = {
                "document_name": doc_name,
                "page_number": result.page_number,
                "section_title": result.section_title,
                "chunk_id": result.chunk_id,
                "relevance_score": result.similarity_score,
                "excerpt": self._create_excerpt(result.content, 100)
            }
            citations.append(citation)
        
        # Assemble context by document
        context_parts = []
        for doc_name, doc_results in doc_sections.items():
            # Sort by page number for logical flow
            doc_results.sort(key=lambda x: x.page_number)
            
            doc_context = f"=== From {doc_name} ===\n"
            for result in doc_results:
                section_info = f" (Page {result.page_number}"
                if result.section_title:
                    section_info += f", {result.section_title}"
                section_info += ")"
                
                doc_context += f"\n{section_info}:\n{result.content}\n"
            
            context_parts.append(doc_context)
        
        context = "\n\n".join(context_parts)
        
        return context, citations
    
    def get_document_coverage(self, query: str) -> Dict[str, int]:
        """
        Get information about which documents contain relevant content for a query.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary mapping document names to number of relevant chunks
        """
        search_results = self.vector_store.search(query, k=20)  # Get more results
        
        coverage = {}
        for result in search_results:
            doc_name = result.document_name
            coverage[doc_name] = coverage.get(doc_name, 0) + 1
        
        return coverage
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance.
        
        Returns:
            Dictionary with retrieval statistics
        """
        avg_processing_time = (
            self.total_processing_time / self.query_count 
            if self.query_count > 0 else 0
        )
        
        return {
            "total_queries": self.query_count,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "max_context_length": self.max_context_length,
            "max_chunks_per_query": self.max_chunks_per_query,
            "min_relevance_score": self.min_relevance_score
        }
    
    def validate_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate that citations point to existing chunks.
        
        Args:
            citations: List of citation dictionaries
            
        Returns:
            List of validated citations
        """
        validated = []
        
        for citation in citations:
            chunk_id = citation.get("chunk_id")
            if chunk_id:
                chunk_data = self.vector_store.get_chunk_by_id(chunk_id)
                if chunk_data:
                    validated.append(citation)
                else:
                    logger.warning(f"Invalid citation: chunk {chunk_id} not found")
        
        return validated


# Example usage and testing
if __name__ == "__main__":
    from vector_store import VectorStore
    from text_processor import ProcessedChunk
    
    # Create sample data for testing
    sample_chunks = [
        ProcessedChunk(
            chunk_id="bio_chunk_001",
            document_name="biology_textbook.pdf",
            page_number=15,
            section_title="Cell Structure",
            content="The cell is the basic unit of life. All living organisms are composed of one or more cells. Cells contain various organelles including the nucleus, mitochondria, and ribosomes.",
            char_start=0,
            char_end=150,
            word_count=25,
            chunk_index=0,
            overlap_with_previous=False,
            overlap_with_next=True
        ),
        ProcessedChunk(
            chunk_id="bio_chunk_002",
            document_name="biology_textbook.pdf",
            page_number=16,
            section_title="Photosynthesis",
            content="Photosynthesis is the process by which plants convert light energy into chemical energy. This process occurs in the chloroplasts and involves the conversion of carbon dioxide and water into glucose.",
            char_start=140,
            char_end=300,
            word_count=30,
            chunk_index=1,
            overlap_with_previous=True,
            overlap_with_next=False
        )
    ]
    
    # Initialize components
    vector_store = VectorStore(
        model_name="all-MiniLM-L6-v2",
        embeddings_dir="./test_embeddings/"
    )
    vector_store.add_chunks(sample_chunks)
    
    retriever = DocumentRetriever(vector_store)
    
    # Test retrieval
    result = retriever.retrieve_context("What is a cell?")
    
    print(f"Query: {result.query}")
    print(f"Retrieved {result.total_chunks} chunks")
    print(f"Retrieval score: {result.retrieval_score:.3f}")
    print(f"Processing time: {result.processing_time_ms}ms")
    print(f"\nContext:\n{result.context}")
    print(f"\nCitations:")
    for i, citation in enumerate(result.citations):
        print(f"{i+1}. {citation['document_name']}, Page {citation['page_number']}")
        print(f"   Relevance: {citation['relevance_score']:.3f}")
        print(f"   Excerpt: {citation['excerpt']}")
    
    # Test summarization retrieval
    summary_result = retriever.retrieve_for_summarization("photosynthesis")
    print(f"\nSummarization context length: {len(summary_result.context)}")
    print(f"Citations for summarization: {len(summary_result.citations)}")
    
    # Print statistics
    stats = retriever.get_retrieval_stats()
    print(f"\nRetrieval Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

