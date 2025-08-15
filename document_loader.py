"""
Document Loader Module for Klaro Academic Chatbot

This module handles PDF document loading, text extraction, and preprocessing
for the RAG pipeline. It provides robust PDF parsing with metadata extraction
and error handling.
"""

import fitz  # PyMuPDF
# from fitz import Document 
from typing import List, Dict, Tuple, Optional, Any  # Add Any if you want

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Metadata structure for processed documents"""
    filename: str
    title: Optional[str]
    subject: Optional[str]
    author: Optional[str]
    page_count: int
    file_size: int
    checksum: str
    creation_date: Optional[str]
    modification_date: Optional[str]

@dataclass
class DocumentChunk:
    """Structure for document text chunks with metadata"""
    document_id: str
    chunk_index: int
    page_number: int
    section_title: Optional[str]
    content: str
    char_start: int
    char_end: int

class DocumentLoader:
    """
    Handles loading and processing of PDF documents for the Klaro system.
    
    Features:
    - Robust PDF text extraction with layout preservation
    - Metadata extraction and validation
    - Error handling for corrupted or unsupported files
    - Duplicate detection via content hashing
    - Progress tracking for batch operations
    """
    
    def __init__(self, docs_folder: str = "./klaro_docs/"):
        """
        Initialize the document loader.
        
        Args:
            docs_folder: Path to the folder containing PDF documents
        """
        self.docs_folder = Path(docs_folder)
        self.docs_folder.mkdir(exist_ok=True)
        self.processed_documents: Dict[str, DocumentMetadata] = {}
        
    def load_single_pdf(self, file_path: Path) -> Tuple[DocumentMetadata, str]:
        """
        Load and extract text from a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (metadata, full_text)
            
        Raises:
            ValueError: If file is not a valid PDF or cannot be processed
        """
        try:
            # Validate file exists and is readable
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
                
            if not file_path.suffix.lower() == '.pdf':
                raise ValueError(f"Not a PDF file: {file_path}")
            
            # Open PDF document
            doc = fitz.open(str(file_path))
            
            # Extract metadata
            metadata = self._extract_metadata(doc, file_path)
            
            # Extract text from all pages
            full_text = self._extract_text_from_pdf(doc)
            
            # Close document
            doc.close()
            
            logger.info(f"Successfully loaded PDF: {file_path.name} ({metadata.page_count} pages)")
            
            return metadata, full_text
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise ValueError(f"Failed to load PDF {file_path}: {str(e)}")
    
    def _extract_metadata(self, doc: Any, file_path: Path) -> DocumentMetadata:
        """Extract metadata from PDF document"""
        try:
            # Get PDF metadata
            pdf_metadata = doc.metadata
            
            # Calculate file checksum
            file_content = file_path.read_bytes()
            checksum = hashlib.sha256(file_content).hexdigest()
            
            # Extract title from metadata or filename
            title = pdf_metadata.get('title', '').strip()
            if not title:
                title = file_path.stem.replace('_', ' ').replace('-', ' ').title()
            
            # Create metadata object
            metadata = DocumentMetadata(
                filename=file_path.name,
                title=title,
                subject=pdf_metadata.get('subject', '').strip() or None,
                author=pdf_metadata.get('author', '').strip() or None,
                page_count=doc.page_count,
                file_size=len(file_content),
                checksum=checksum,
                creation_date=pdf_metadata.get('creationDate', '').strip() or None,
                modification_date=pdf_metadata.get('modDate', '').strip() or None
            )
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {str(e)}")
            # Return minimal metadata if extraction fails
            return DocumentMetadata(
                filename=file_path.name,
                title=file_path.stem,
                subject=None,
                author=None,
                page_count=doc.page_count,
                file_size=file_path.stat().st_size,
                checksum="unknown",
                creation_date=None,
                modification_date=None
            )
    
    def _extract_text_from_pdf(self, doc: Any) -> str:
        """
        Extract text from all pages of a PDF document.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Full text content of the document
        """
        full_text = []
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # Clean up text
                text = self._clean_extracted_text(text)
                
                if text.strip():  # Only add non-empty pages
                    # Add page marker for citation purposes
                    page_header = f"\n--- Page {page_num + 1} ---\n"
                    full_text.append(page_header + text)
                    
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        return "\n\n".join(full_text)
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip likely headers/footers (very short lines with numbers)
            if len(line) < 5 and line.isdigit():
                continue
                
            # Skip lines that are just page numbers or common headers
            if re.match(r'^(page\s+)?\d+$', line.lower()):
                continue
                
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def load_all_pdfs(self) -> Dict[str, Tuple[DocumentMetadata, str]]:
        """
        Load all PDF files from the documents folder.
        
        Returns:
            Dictionary mapping filename to (metadata, text) tuples
        """
        documents = {}
        pdf_files = [f for f in self.docs_folder.iterdir() if f.suffix.lower() == ".pdf"]
        
        print("Files in folder:", list(self.docs_folder.iterdir()))
        print("PDF files found:", pdf_files)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.docs_folder}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for file_path in pdf_files:
            try:
                metadata, text = self.load_single_pdf(file_path)
                documents[file_path.name] = (metadata, text)
                self.processed_documents[file_path.name] = metadata
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed {len(documents)} PDF files")
        return documents
    
    def get_document_info(self, filename: str) -> Optional[DocumentMetadata]:
        """
        Get metadata for a specific document.
        
        Args:
            filename: Name of the document file
            
        Returns:
            Document metadata or None if not found
        """
        return self.processed_documents.get(filename)
    
    def validate_document(self, file_path: Path) -> bool:
        """
        Validate that a file is a readable PDF document.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            if not file_path.exists() or not file_path.suffix.lower() == '.pdf':
                return False
                
            # Try to open the PDF
            doc = fitz.open(str(file_path))
            page_count = doc.page_count
            doc.close()
            
            return page_count > 0
            
        except Exception:
            return False
    
    def get_processing_stats(self) -> Dict[str, int]:
        """
        Get statistics about processed documents.
        
        Returns:
            Dictionary with processing statistics
        """
        if not self.processed_documents:
            return {"total_documents": 0, "total_pages": 0, "total_size_mb": 0, "average_pages_per_doc": 0}
        
        total_pages = sum(doc.page_count for doc in self.processed_documents.values())
        total_size = sum(doc.file_size for doc in self.processed_documents.values())
        
        return {
            "total_documents": len(self.processed_documents),
            "total_pages": total_pages,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "average_pages_per_doc": round(total_pages / len(self.processed_documents), 1)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize document loader
    loader = DocumentLoader("./klaro_docs/")
    
    # Load all PDFs
    documents = loader.load_all_pdfs()
    
    # Print processing statistics
    stats = loader.get_processing_stats()
    print(f"Processing Statistics:")
    print(f"- Total Documents: {stats['total_documents']}")
    print(f"- Total Pages: {stats['total_pages']}")
    print(f"- Total Size: {stats['total_size_mb']} MB")
    print(f"- Average Pages per Document: {stats['average_pages_per_doc']}")
    
    # Print document summaries
    for filename, (metadata, text) in documents.items():
        print(f"\nDocument: {filename}")
        print(f"- Title: {metadata.title}")
        print(f"- Pages: {metadata.page_count}")
        print(f"- Text Length: {len(text)} characters")
        print(f"- Preview: {text[:200]}...")

