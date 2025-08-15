"""
Text Processing Module for Klaro Academic Chatbot

This module handles text chunking, preprocessing, and preparation for vector
embedding. It implements intelligent chunking strategies optimized for
educational content and maintains proper citation tracking.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedChunk:
    """Structure for processed text chunks with enhanced metadata"""
    chunk_id: str
    document_name: str
    page_number: int
    section_title: Optional[str]
    content: str
    char_start: int
    char_end: int
    word_count: int
    chunk_index: int
    overlap_with_previous: bool
    overlap_with_next: bool

class TextProcessor:
    """
    Handles text preprocessing, chunking, and preparation for vector embedding.
    
    Features:
    - Intelligent chunking with context preservation
    - Section-aware splitting for academic content
    - Citation metadata tracking
    - Overlap management for context continuity
    - Content quality filtering
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 min_chunk_size: int = 100):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum chunk size to avoid very small fragments
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize the text splitter with academic content optimizations
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation sentences
                "? ",    # Question sentences
                "; ",    # Semicolon breaks
                ", ",    # Comma breaks
                " ",     # Word breaks
                ""       # Character breaks (last resort)
            ]
        )
        
        # Patterns for detecting academic content structure
        self.section_patterns = [
            r'^(Chapter|Section|Part)\s+\d+',
            r'^\d+\.\d+\s+',  # Numbered sections like "1.1 Introduction"
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headings
            r'^\*\*[^*]+\*\*$',  # Bold headings
            r'^#{1,6}\s+',  # Markdown headings
        ]
        
        self.processed_chunks: List[ProcessedChunk] = []
    
    def process_document(self, 
                        document_name: str, 
                        text: str) -> List[ProcessedChunk]:
        """
        Process a document into chunks with metadata.
        
        Args:
            document_name: Name of the source document
            text: Full text content of the document
            
        Returns:
            List of processed chunks with metadata
        """
        logger.info(f"Processing document: {document_name}")
        
        # Clean and preprocess the text
        cleaned_text = self._preprocess_text(text)
        
        # Extract page information
        page_info = self._extract_page_info(cleaned_text)
        
        # Create LangChain document
        doc = Document(
            page_content=cleaned_text,
            metadata={"source": document_name}
        )
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Process chunks with enhanced metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunk = self._create_processed_chunk(
                chunk, document_name, i, page_info
            )
            
            # Filter out low-quality chunks
            if self._is_valid_chunk(processed_chunk):
                processed_chunks.append(processed_chunk)
        
        logger.info(f"Created {len(processed_chunks)} chunks from {document_name}")
        self.processed_chunks.extend(processed_chunks)
        
        return processed_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for optimal chunking.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common OCR errors in academic texts
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
        text = re.sub(r'(\w)(\d)', r'\1 \2', text)  # Word-number separation
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Number-word separation
        
        # Normalize quotation marks
        # text = re.sub(r'["""]', '"', text)
        # text = re.sub(r'[''']', "'", text)
        
        text = re.sub(r"[“”«»„‟]", '"', text)  # Normalize all fancy double quotes to "
        text = re.sub(r"[‘’‚‛]", "'", text)    # Normalize all fancy single quotes to '
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Normalize section headers
        text = re.sub(r'\n([A-Z][A-Z\s]+)\n', r'\n\n\1\n\n', text)
        
        return text.strip()
    
    def _extract_page_info(self, text: str) -> Dict[int, Tuple[int, int]]:
        """
        Extract page number information from text.
        
        Args:
            text: Text with page markers
            
        Returns:
            Dictionary mapping page numbers to (start_pos, end_pos) in text
        """
        page_info = {}
        page_pattern = r'--- Page (\d+) ---'
        
        matches = list(re.finditer(page_pattern, text))
        
        for i, match in enumerate(matches):
            page_num = int(match.group(1))
            start_pos = match.end()
            
            # End position is start of next page or end of text
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            page_info[page_num] = (start_pos, end_pos)
        
        return page_info
    
    def _create_processed_chunk(self, 
                               chunk: Document, 
                               document_name: str, 
                               chunk_index: int,
                               page_info: Dict[int, Tuple[int, int]]) -> ProcessedChunk:
        """
        Create a ProcessedChunk with enhanced metadata.
        
        Args:
            chunk: LangChain document chunk
            document_name: Source document name
            chunk_index: Index of this chunk in the document
            page_info: Page position information
            
        Returns:
            ProcessedChunk with metadata
        """
        content = chunk.page_content
        
        # Find which page this chunk belongs to
        page_number = self._find_page_number(content, page_info)
        
        # Extract section title if present
        section_title = self._extract_section_title(content)
        
        # Calculate positions (simplified for this implementation)
        char_start = chunk_index * (self.chunk_size - self.chunk_overlap)
        char_end = char_start + len(content)
        
        # Generate unique chunk ID
        chunk_id = f"{document_name}_chunk_{chunk_index:04d}"
        
        # Count words
        word_count = len(content.split())
        
        return ProcessedChunk(
            chunk_id=chunk_id,
            document_name=document_name,
            page_number=page_number,
            section_title=section_title,
            content=content,
            char_start=char_start,
            char_end=char_end,
            word_count=word_count,
            chunk_index=chunk_index,
            overlap_with_previous=chunk_index > 0,
            overlap_with_next=True  # Will be updated when processing next chunk
        )
    
    def _find_page_number(self, 
                         content: str, 
                         page_info: Dict[int, Tuple[int, int]]) -> int:
        """
        Find the page number for a given chunk content.
        
        Args:
            content: Chunk content
            page_info: Page position information
            
        Returns:
            Page number (1-based)
        """
        # Look for page markers in the content
        page_match = re.search(r'--- Page (\d+) ---', content)
        if page_match:
            return int(page_match.group(1))
        
        # If no direct page marker, estimate based on content position
        # This is a simplified approach - in practice, you'd want more
        # sophisticated position tracking
        return 1
    
    def _extract_section_title(self, content: str) -> Optional[str]:
        """
        Extract section title from chunk content.
        
        Args:
            content: Chunk content
            
        Returns:
            Section title if found, None otherwise
        """
        lines = content.split('\n')
        
        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            
            # Check against section patterns
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return line
            
            # Check for lines that look like headings
            if (len(line) < 100 and 
                len(line) > 5 and 
                not line.endswith('.') and
                any(c.isupper() for c in line)):
                return line
        
        return None
    
    def _is_valid_chunk(self, chunk: ProcessedChunk) -> bool:
        """
        Validate chunk quality and relevance.
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            True if chunk is valid, False otherwise
        """
        content = chunk.content.strip()
        
        # Check minimum size
        if len(content) < self.min_chunk_size:
            return False
        
        # Check if chunk is mostly whitespace or special characters
        if len(re.sub(r'[^a-zA-Z0-9]', '', content)) < self.min_chunk_size // 2:
            return False
        
        # Check if chunk contains meaningful content (has some words)
        if chunk.word_count < 10:
            return False
        
        # Filter out chunks that are mostly page headers/footers
        if self._is_header_footer(content):
            return False
        
        return True
    
    def _is_header_footer(self, content: str) -> bool:
        """
        Check if content is likely a header or footer.
        
        Args:
            content: Content to check
            
        Returns:
            True if likely header/footer, False otherwise
        """
        content = content.strip().lower()
        
        # Common header/footer patterns
        header_footer_patterns = [
            r'^page \d+$',
            r'^\d+$',
            r'^chapter \d+$',
            r'^table of contents$',
            r'^index$',
            r'^bibliography$',
            r'^references$',
        ]
        
        for pattern in header_footer_patterns:
            if re.match(pattern, content):
                return True
        
        # Check if content is very short and repetitive
        if len(content) < 50 and len(set(content.split())) < 5:
            return True
        
        return False
    
    def get_chunks_by_document(self, document_name: str) -> List[ProcessedChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of chunks for the document
        """
        return [chunk for chunk in self.processed_chunks 
                if chunk.document_name == document_name]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[ProcessedChunk]:
        """
        Get a specific chunk by its ID.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            Chunk if found, None otherwise
        """
        for chunk in self.processed_chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get statistics about processed chunks.
        
        Returns:
            Dictionary with processing statistics
        """
        if not self.processed_chunks:
            return {
                "total_chunks": 0,
                "total_words": 0,
                "average_chunk_size": 0,
                "documents_processed": 0
            }
        
        total_words = sum(chunk.word_count for chunk in self.processed_chunks)
        total_chars = sum(len(chunk.content) for chunk in self.processed_chunks)
        unique_docs = len(set(chunk.document_name for chunk in self.processed_chunks))
        
        return {
            "total_chunks": len(self.processed_chunks),
            "total_words": total_words,
            "total_characters": total_chars,
            "average_chunk_size": round(total_chars / len(self.processed_chunks), 1),
            "average_words_per_chunk": round(total_words / len(self.processed_chunks), 1),
            "documents_processed": unique_docs,
            "chunks_per_document": round(len(self.processed_chunks) / unique_docs, 1)
        }
    
    def export_chunks_for_embedding(self) -> List[Dict[str, any]]:
        """
        Export chunks in format suitable for embedding generation.
        
        Returns:
            List of dictionaries with chunk data
        """
        return [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.content,
                "metadata": {
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "word_count": chunk.word_count
                }
            }
            for chunk in self.processed_chunks
        ]


# Example usage and testing
if __name__ == "__main__":
    # Sample text for testing
    sample_text = """
    --- Page 1 ---
    
    Chapter 1: Introduction to Biology
    
    Biology is the scientific study of life and living organisms. It encompasses
    a wide range of topics from molecular biology to ecology. This chapter will
    introduce the fundamental concepts that form the foundation of biological
    sciences.
    
    1.1 What is Life?
    
    Life is characterized by several key properties including organization,
    metabolism, growth, adaptation, response to stimuli, and reproduction.
    These characteristics distinguish living organisms from non-living matter.
    
    --- Page 2 ---
    
    1.2 Levels of Organization
    
    Biological organization exists at multiple levels, from atoms and molecules
    to ecosystems and the biosphere. Each level has emergent properties that
    arise from the interactions of its components.
    
    The hierarchy includes: atoms, molecules, organelles, cells, tissues,
    organs, organ systems, organisms, populations, communities, ecosystems,
    and the biosphere.
    """
    
    # Initialize processor
    processor = TextProcessor(chunk_size=300, chunk_overlap=50)
    
    # Process the sample text
    chunks = processor.process_document("sample_biology.pdf", sample_text)
    
    # Print results
    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Section: {chunk.section_title}")
        print(f"Words: {chunk.word_count}")
        print(f"Content: {chunk.content[:100]}...")
    
    # Print statistics
    stats = processor.get_processing_stats()
    print(f"\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")

