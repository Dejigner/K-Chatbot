"""
Summarizer Module for Klaro Academic Chatbot

This module handles topic summarization across multiple documents, providing
comprehensive overviews of academic concepts with proper source attribution
and educational structuring.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from retriever import DocumentRetriever, RetrievalResult
from llm_interface import LLMInterface, LLMResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """Structure for summarization results"""
    topic: str
    summary_text: str
    sources_used: List[Dict[str, Any]]
    coverage_score: float
    quality_score: float
    word_count: int
    processing_time_ms: int
    retrieval_stats: Dict[str, Any]
    llm_stats: Dict[str, Any]

@dataclass
class TopicCoverage:
    """Structure for topic coverage analysis"""
    topic: str
    documents_with_content: Dict[str, int]  # doc_name -> chunk_count
    total_relevant_chunks: int
    coverage_percentage: float
    key_sections: List[str]

class TopicSummarizer:
    """
    Handles comprehensive topic summarization across multiple documents.
    
    Features:
    - Multi-document topic analysis and synthesis
    - Educational content structuring
    - Source diversity optimization
    - Quality assessment and validation
    - Coverage analysis for curriculum alignment
    """
    
    def __init__(self, 
                 retriever: DocumentRetriever,
                 llm_interface: LLMInterface,
                 min_sources: int = 2,
                 max_summary_length: int = 1500):
        """
        Initialize the topic summarizer.
        
        Args:
            retriever: Document retriever instance
            llm_interface: LLM interface for text generation
            min_sources: Minimum number of sources required
            max_summary_length: Maximum length for summaries
        """
        self.retriever = retriever
        self.llm_interface = llm_interface
        self.min_sources = min_sources
        self.max_summary_length = max_summary_length
        
        # Summarization statistics
        self.summaries_generated = 0
        self.total_processing_time = 0
        self.topic_coverage_cache: Dict[str, TopicCoverage] = {}
    
    def summarize_topic(self, 
                       topic: str,
                       max_sources: int = 10,
                       require_multiple_docs: bool = True) -> SummaryResult:
        """
        Generate a comprehensive summary of a topic.
        
        Args:
            topic: Topic to summarize
            max_sources: Maximum number of source chunks to use
            require_multiple_docs: Whether to require content from multiple documents
            
        Returns:
            SummaryResult with comprehensive topic summary
        """
        import time
        start_time = time.time()
        
        logger.info(f"Generating summary for topic: '{topic}'")
        
        # Analyze topic coverage first
        coverage = self.analyze_topic_coverage(topic)
        
        if coverage.total_relevant_chunks == 0:
            logger.warning(f"No relevant content found for topic: '{topic}'")
            return self._create_empty_summary_result(topic, int((time.time() - start_time) * 1000))
        
        # Check if we have sufficient source diversity
        if require_multiple_docs and len(coverage.documents_with_content) < self.min_sources:
            logger.warning(f"Insufficient source diversity for topic '{topic}': {len(coverage.documents_with_content)} documents")
        
        # Retrieve context optimized for summarization
        retrieval_result = self.retriever.retrieve_for_summarization(
            topic, 
            max_chunks=max_sources
        )
        
        if not retrieval_result.context.strip():
            logger.warning(f"No context retrieved for topic: '{topic}'")
            return self._create_empty_summary_result(topic, int((time.time() - start_time) * 1000))
        
        # Generate summary using LLM
        llm_response = self.llm_interface.summarize_topic(topic, retrieval_result.context)
        
        # Validate and enhance the summary
        enhanced_summary = self._enhance_summary(
            llm_response.response_text, 
            retrieval_result.citations,
            topic
        )
        
        # Calculate quality scores
        coverage_score = self._calculate_coverage_score(coverage, retrieval_result)
        quality_score = self._calculate_quality_score(enhanced_summary, retrieval_result)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Update statistics
        self.summaries_generated += 1
        self.total_processing_time += processing_time
        
        result = SummaryResult(
            topic=topic,
            summary_text=enhanced_summary,
            sources_used=retrieval_result.citations,
            coverage_score=coverage_score,
            quality_score=quality_score,
            word_count=len(enhanced_summary.split()),
            processing_time_ms=processing_time,
            retrieval_stats={
                "chunks_retrieved": retrieval_result.total_chunks,
                "retrieval_score": retrieval_result.retrieval_score,
                "retrieval_time_ms": retrieval_result.processing_time_ms
            },
            llm_stats={
                "inference_time_ms": llm_response.inference_time_ms,
                "tokens_generated": llm_response.completion_tokens,
                "model_name": llm_response.model_name
            }
        )
        
        logger.info(f"Generated summary for '{topic}' in {processing_time}ms ({result.word_count} words)")
        return result
    
    def analyze_topic_coverage(self, topic: str) -> TopicCoverage:
        """
        Analyze how well a topic is covered across available documents.
        
        Args:
            topic: Topic to analyze
            
        Returns:
            TopicCoverage with analysis results
        """
        # Check cache first
        if topic in self.topic_coverage_cache:
            return self.topic_coverage_cache[topic]
        
        logger.info(f"Analyzing coverage for topic: '{topic}'")
        
        # Get document coverage from retriever
        doc_coverage = self.retriever.get_document_coverage(topic)
        
        # Calculate total relevant chunks
        total_chunks = sum(doc_coverage.values())
        
        # Calculate coverage percentage (simplified metric)
        # This could be enhanced with more sophisticated analysis
        max_possible_coverage = len(doc_coverage) * 5  # Assume max 5 relevant chunks per doc
        coverage_percentage = min(100.0, (total_chunks / max_possible_coverage) * 100) if max_possible_coverage > 0 else 0.0
        
        # Extract key sections (simplified - could be enhanced)
        key_sections = self._extract_key_sections(topic, doc_coverage)
        
        coverage = TopicCoverage(
            topic=topic,
            documents_with_content=doc_coverage,
            total_relevant_chunks=total_chunks,
            coverage_percentage=coverage_percentage,
            key_sections=key_sections
        )
        
        # Cache the result
        self.topic_coverage_cache[topic] = coverage
        
        return coverage
    
    def _extract_key_sections(self, topic: str, doc_coverage: Dict[str, int]) -> List[str]:
        """
        Extract key sections related to a topic.
        
        Args:
            topic: Topic being analyzed
            doc_coverage: Document coverage information
            
        Returns:
            List of key section titles
        """
        # This is a simplified implementation
        # In practice, you'd want to analyze actual section titles from retrieved chunks
        key_sections = []
        
        # Get some sample chunks to analyze section titles
        search_results = self.retriever.vector_store.search(topic, k=10)
        
        section_titles = set()
        for result in search_results:
            if result.section_title:
                section_titles.add(result.section_title)
        
        return list(section_titles)[:5]  # Return top 5 sections
    
    def _enhance_summary(self, 
                        summary_text: str, 
                        citations: List[Dict[str, Any]],
                        topic: str) -> str:
        """
        Enhance the generated summary with additional formatting and validation.
        
        Args:
            summary_text: Original summary text
            citations: List of citations
            topic: Topic being summarized
            
        Returns:
            Enhanced summary text
        """
        enhanced = summary_text.strip()
        
        # Add topic header if not present
        if not enhanced.lower().startswith(topic.lower()):
            enhanced = f"# {topic.title()}\n\n{enhanced}"
        
        # Ensure proper paragraph structure
        enhanced = self._format_paragraphs(enhanced)
        
        # Add source summary at the end if not present
        if not self._has_source_summary(enhanced):
            enhanced = self._add_source_summary(enhanced, citations)
        
        # Validate length and truncate if necessary
        if len(enhanced) > self.max_summary_length:
            enhanced = self._truncate_summary(enhanced, self.max_summary_length)
        
        return enhanced
    
    def _format_paragraphs(self, text: str) -> str:
        """Format text into proper paragraphs."""
        # Split into sentences and group into paragraphs
        sentences = text.split('. ')
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            
            # Start new paragraph after 3-4 sentences or at logical breaks
            if (len(current_paragraph) >= 3 or 
                any(keyword in sentence.lower() for keyword in ['however', 'furthermore', 'additionally', 'in contrast'])):
                paragraphs.append('. '.join(current_paragraph) + ('.' if not sentence.endswith('.') else ''))
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append('. '.join(current_paragraph) + ('.' if not text.endswith('.') else ''))
        
        return '\n\n'.join(paragraphs)
    
    def _has_source_summary(self, text: str) -> bool:
        """Check if text already has a source summary section."""
        return any(keyword in text.lower() for keyword in ['sources:', 'references:', 'based on:'])
    
    def _add_source_summary(self, text: str, citations: List[Dict[str, Any]]) -> str:
        """Add a source summary section to the text."""
        if not citations:
            return text
        
        # Group citations by document
        docs_cited = {}
        for citation in citations:
            doc_name = citation['document_name']
            if doc_name not in docs_cited:
                docs_cited[doc_name] = []
            docs_cited[doc_name].append(citation['page_number'])
        
        # Create source summary
        source_lines = []
        for doc_name, pages in docs_cited.items():
            pages = sorted(set(pages))  # Remove duplicates and sort
            if len(pages) == 1:
                source_lines.append(f"- {doc_name}, Page {pages[0]}")
            elif len(pages) <= 3:
                source_lines.append(f"- {doc_name}, Pages {', '.join(map(str, pages))}")
            else:
                source_lines.append(f"- {doc_name}, Pages {pages[0]}-{pages[-1]} (and others)")
        
        source_summary = "\n\n**Sources:**\n" + "\n".join(source_lines)
        
        return text + source_summary
    
    def _truncate_summary(self, text: str, max_length: int) -> str:
        """Truncate summary while preserving structure."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at paragraph boundary
        paragraphs = text.split('\n\n')
        truncated = ""
        
        for paragraph in paragraphs:
            if len(truncated + paragraph) <= max_length - 50:  # Leave room for truncation notice
                truncated += paragraph + "\n\n"
            else:
                break
        
        # Add truncation notice
        truncated = truncated.strip() + "\n\n[Summary truncated for length]"
        
        return truncated
    
    def _calculate_coverage_score(self, 
                                 coverage: TopicCoverage, 
                                 retrieval_result: RetrievalResult) -> float:
        """
        Calculate how well the topic is covered by available sources.
        
        Args:
            coverage: Topic coverage analysis
            retrieval_result: Retrieval results
            
        Returns:
            Coverage score (0.0 to 1.0)
        """
        # Base score from coverage percentage
        base_score = coverage.coverage_percentage / 100.0
        
        # Bonus for source diversity
        diversity_bonus = min(0.2, len(coverage.documents_with_content) * 0.05)
        
        # Bonus for retrieval quality
        retrieval_bonus = retrieval_result.retrieval_score * 0.1
        
        # Penalty for insufficient sources
        if len(coverage.documents_with_content) < self.min_sources:
            source_penalty = 0.2
        else:
            source_penalty = 0.0
        
        final_score = min(1.0, max(0.0, base_score + diversity_bonus + retrieval_bonus - source_penalty))
        
        return final_score
    
    def _calculate_quality_score(self, 
                                summary_text: str, 
                                retrieval_result: RetrievalResult) -> float:
        """
        Calculate the quality score of the generated summary.
        
        Args:
            summary_text: Generated summary text
            retrieval_result: Retrieval results used
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 0.0
        
        # Length appropriateness (0.2 points)
        word_count = len(summary_text.split())
        if 100 <= word_count <= 500:
            score += 0.2
        elif 50 <= word_count < 100 or 500 < word_count <= 800:
            score += 0.1
        
        # Citation presence (0.3 points)
        citation_count = summary_text.count('[Source:')
        if citation_count >= 3:
            score += 0.3
        elif citation_count >= 1:
            score += 0.2
        
        # Structure quality (0.2 points)
        if '\n\n' in summary_text:  # Has paragraphs
            score += 0.1
        if any(header in summary_text for header in ['#', '**', 'Sources:']):  # Has structure
            score += 0.1
        
        # Content coherence (0.3 points) - simplified check
        sentences = summary_text.split('.')
        if len(sentences) >= 5:  # Has multiple sentences
            score += 0.1
        if any(connector in summary_text.lower() for connector in ['however', 'furthermore', 'additionally', 'therefore']):
            score += 0.1
        if not any(issue in summary_text.lower() for issue in ['i think', 'i believe', 'in my opinion']):
            score += 0.1
        
        return min(1.0, score)
    
    def _create_empty_summary_result(self, topic: str, processing_time: int) -> SummaryResult:
        """Create an empty summary result for topics with no content."""
        return SummaryResult(
            topic=topic,
            summary_text=f"I couldn't find sufficient information about '{topic}' in the provided materials to create a comprehensive summary.",
            sources_used=[],
            coverage_score=0.0,
            quality_score=0.0,
            word_count=0,
            processing_time_ms=processing_time,
            retrieval_stats={"chunks_retrieved": 0, "retrieval_score": 0.0, "retrieval_time_ms": 0},
            llm_stats={"inference_time_ms": 0, "tokens_generated": 0, "model_name": "none"}
        )
    
    def batch_summarize_topics(self, topics: List[str]) -> List[SummaryResult]:
        """
        Generate summaries for multiple topics in batch.
        
        Args:
            topics: List of topics to summarize
            
        Returns:
            List of summary results
        """
        logger.info(f"Batch summarizing {len(topics)} topics")
        
        results = []
        for i, topic in enumerate(topics):
            logger.info(f"Processing topic {i+1}/{len(topics)}: {topic}")
            
            try:
                result = self.summarize_topic(topic)
                results.append(result)
            except Exception as e:
                logger.error(f"Error summarizing topic '{topic}': {str(e)}")
                # Create error result
                error_result = self._create_empty_summary_result(topic, 0)
                error_result.summary_text = f"Error generating summary for '{topic}': {str(e)}"
                results.append(error_result)
        
        logger.info(f"Completed batch summarization: {len(results)} results")
        return results
    
    def get_topic_suggestions(self, document_name: Optional[str] = None) -> List[str]:
        """
        Get suggested topics based on available content.
        
        Args:
            document_name: Optional document to focus on
            
        Returns:
            List of suggested topics
        """
        # This is a simplified implementation
        # In practice, you'd analyze section titles, keywords, etc.
        
        if document_name:
            chunks = self.retriever.vector_store.get_chunks_by_document(document_name)
        else:
            chunks = self.retriever.vector_store.chunk_metadata
        
        # Extract potential topics from section titles
        topics = set()
        for chunk in chunks:
            section_title = chunk.get('section_title')
            if section_title:
                # Simple topic extraction from section titles
                cleaned_title = section_title.lower().strip()
                if len(cleaned_title) > 5 and not cleaned_title.isdigit():
                    topics.add(section_title)
        
        return sorted(list(topics))[:20]  # Return top 20 suggestions
    
    def get_summarization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about summarization performance.
        
        Returns:
            Dictionary with summarization statistics
        """
        avg_processing_time = (
            self.total_processing_time / self.summaries_generated 
            if self.summaries_generated > 0 else 0
        )
        
        return {
            "summaries_generated": self.summaries_generated,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "topics_analyzed": len(self.topic_coverage_cache),
            "min_sources_required": self.min_sources,
            "max_summary_length": self.max_summary_length
        }


# Example usage and testing
if __name__ == "__main__":
    from vector_store import VectorStore
    from text_processor import ProcessedChunk
    from llm_interface import MockLLMInterface
    
    # Create sample data
    sample_chunks = [
        ProcessedChunk(
            chunk_id="bio_001",
            document_name="biology_textbook.pdf",
            page_number=10,
            section_title="Introduction to Biology",
            content="Biology is the scientific study of life and living organisms. It encompasses many specialized fields including molecular biology, genetics, ecology, and evolutionary biology.",
            char_start=0, char_end=150, word_count=25, chunk_index=0,
            overlap_with_previous=False, overlap_with_next=True
        ),
        ProcessedChunk(
            chunk_id="bio_002",
            document_name="biology_textbook.pdf",
            page_number=15,
            section_title="Cell Structure",
            content="Cells are the fundamental units of life. They contain various organelles including the nucleus, mitochondria, ribosomes, and endoplasmic reticulum, each with specific functions.",
            char_start=140, char_end=290, word_count=28, chunk_index=1,
            overlap_with_previous=True, overlap_with_next=True
        ),
        ProcessedChunk(
            chunk_id="chem_001",
            document_name="chemistry_basics.pdf",
            page_number=5,
            section_title="Atoms and Molecules",
            content="Atoms are the basic building blocks of matter. They combine to form molecules through chemical bonds. Understanding atomic structure is crucial for biology.",
            char_start=0, char_end=140, word_count=24, chunk_index=0,
            overlap_with_previous=False, overlap_with_next=True
        )
    ]
    
    # Initialize components
    vector_store = VectorStore(
        model_name="all-MiniLM-L6-v2",
        embeddings_dir="./test_embeddings/"
    )
    vector_store.add_chunks(sample_chunks)
    
    retriever = DocumentRetriever(vector_store)
    llm_interface = MockLLMInterface()
    
    summarizer = TopicSummarizer(retriever, llm_interface)
    
    # Test topic summarization
    topic = "cell structure"
    result = summarizer.summarize_topic(topic)
    
    print(f"Topic: {result.topic}")
    print(f"Summary ({result.word_count} words):")
    print(result.summary_text)
    print(f"\nCoverage Score: {result.coverage_score:.2f}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Processing Time: {result.processing_time_ms}ms")
    print(f"Sources Used: {len(result.sources_used)}")
    
    # Test coverage analysis
    coverage = summarizer.analyze_topic_coverage("biology")
    print(f"\nTopic Coverage for 'biology':")
    print(f"- Documents with content: {len(coverage.documents_with_content)}")
    print(f"- Total relevant chunks: {coverage.total_relevant_chunks}")
    print(f"- Coverage percentage: {coverage.coverage_percentage:.1f}%")
    
    # Test topic suggestions
    suggestions = summarizer.get_topic_suggestions()
    print(f"\nTopic Suggestions: {suggestions}")
    
    # Print statistics
    stats = summarizer.get_summarization_stats()
    print(f"\nSummarization Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

