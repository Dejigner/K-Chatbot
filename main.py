"""
Main Application Module for Klaro Academic Chatbot

This module serves as the main entry point for the Klaro system, integrating
all components into a cohesive RAG pipeline for educational Q&A and summarization.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import Klaro components
from document_loader import DocumentLoader
from text_processor import TextProcessor
from vector_store import VectorStore
from retriever import DocumentRetriever
from llm_interface import LLMInterface, MockLLMInterface
from summarizer import TopicSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('klaro.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KlaroSystem:
    """
    Main Klaro Academic Chatbot System.
    
    Integrates all components to provide:
    - Document processing and indexing
    - Question answering with citations
    - Topic summarization across documents
    - System management and statistics
    """
    
    def __init__(self, 
                #  docs_folder: str = "./klaro_docs/",
                #  embeddings_dir: str = "./embeddings/",
                #  models_dir: str = "./models/",
                #  embedding_model: str = "hkunlp/instructor-xl",
                #  llm_model_path: Optional[str] = None,
                #  use_mock_llm: bool = False):
        
                docs_folder: str = "./klaro_docs/",
                embeddings_dir: str = "./embeddings/",
                models_dir: str = "./models/",
                embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",  # smaller, ~100MB
                llm_model_path: Optional[str] = None,
                use_mock_llm: bool = False):
        """
        Initialize the Klaro system.
        
        Args:
            docs_folder: Path to PDF documents folder
            embeddings_dir: Path to embeddings storage
            models_dir: Path to LLM models
            embedding_model: Name of embedding model to use
            llm_model_path: Path to local LLM model file
            use_mock_llm: Whether to use mock LLM for testing
        """
        self.docs_folder = Path(docs_folder)
        self.embeddings_dir = Path(embeddings_dir)
        self.models_dir = Path(models_dir)
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"  # ensure fallback
        self.llm_model_path = llm_model_path
        self.use_mock_llm = use_mock_llm
        
        # Create directories
        self.docs_folder.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.document_loader = None
        self.text_processor = None
        self.vector_store = None
        self.retriever = None
        self.llm_interface = None
        self.summarizer = None
        
        # System state
        self.initialized = False
        self.documents_loaded = False
        
        logger.info("Klaro system initialized")
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Klaro system components...")
            
            # Initialize document loader
            self.document_loader = DocumentLoader(str(self.docs_folder))
            
            # Initialize text processor
            self.text_processor = TextProcessor(
                chunk_size=500,
                chunk_overlap=50,
                min_chunk_size=100
            )
            
            # Initialize vector store
            self.vector_store = VectorStore(
                model_name=self.embedding_model,
                embeddings_dir=str(self.embeddings_dir),
                index_name="klaro_index"
            )
            
            # Initialize retriever
            self.retriever = DocumentRetriever(
                vector_store=self.vector_store,
                max_context_length=4000,
                max_chunks_per_query=5,
                min_relevance_score=0.1
            )
            
            # Initialize LLM interface
            if not self.llm_model_path:  # require real model
                logger.error("No LLM model path provided. Download the model and pass --llm-model <path>")
                return False
            else:
                self.use_mock_llm = False  # ensure real model
                logger.info(f"Loading LLM model: {self.llm_model_path}")
                self.llm_interface = LLMInterface(
                     model_path=self.llm_model_path,
                    model_name="mistral-7b-instruct",  # real local Mistral
                     max_tokens=512,
                     temperature=0.1,
                     context_window=4096
                 )
                 # Load the model
                if not self.llm_interface.load_model():
                     logger.error("Failed to load LLM model")
                     return False
            
            # Initialize summarizer
            self.summarizer = TopicSummarizer(
                retriever=self.retriever,
                llm_interface=self.llm_interface,
                min_sources=2,
                max_summary_length=1500
            )
            
            self.initialized = True
            logger.info("Klaro system initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Klaro system: {str(e)}")
            return False
    
    def load_documents(self, force_reload: bool = False) -> bool:
        """
        Load and process all documents in the docs folder.
        
        Args:
            force_reload: Whether to force reloading even if already loaded
            
        Returns:
            True if documents loaded successfully, False otherwise
        """
        if not self.initialized:
            logger.error("System not initialized. Call initialize() first.")
            return False
        
        if self.documents_loaded and not force_reload:
            logger.info("Documents already loaded. Use force_reload=True to reload.")
            return True
        
        try:
            logger.info("Loading documents...")
            
            # Load PDFs
            documents = self.document_loader.load_all_pdfs()
            
            if not documents:
                logger.warning("No documents found to load")
                return False
            
            # Process documents into chunks
            all_chunks = []
            for filename, (metadata, text) in documents.items():
                logger.info(f"Processing document: {filename}")
                chunks = self.text_processor.process_document(filename, text)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                logger.error("No chunks generated from documents")
                return False
            
            # Add chunks to vector store
            logger.info(f"Adding {len(all_chunks)} chunks to vector store...")
            self.vector_store.add_chunks(all_chunks)
            
            self.documents_loaded = True
            logger.info(f"Successfully loaded {len(documents)} documents with {len(all_chunks)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question based on loaded documents.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.documents_loaded:
            return {
                "success": False,
                "error": "No documents loaded. Please load documents first.",
                "answer": "",
                "citations": [],
                "stats": {}
            }
        
        try:
            logger.info(f"Processing question: '{question[:50]}...'")
            
            # Retrieve relevant context
            retrieval_result = self.retriever.retrieve_context(question)
            
            if not retrieval_result.context.strip():
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in the provided materials to answer your question.",
                    "citations": [],
                    "stats": {
                        "retrieval_time_ms": retrieval_result.processing_time_ms,
                        "chunks_found": 0,
                        "inference_time_ms": 0
                    }
                }
            
            # Generate answer using LLM
            llm_response = self.llm_interface.answer_question(question, retrieval_result.context)
            
            # Validate response
            is_valid, issues = self.llm_interface.validate_response(
                llm_response.response_text, 
                retrieval_result.context
            )
            
            if not is_valid:
                logger.warning(f"Response validation issues: {issues}")
            
            return {
                "success": True,
                "answer": llm_response.response_text,
                "citations": retrieval_result.citations,
                "validation": {
                    "is_valid": is_valid,
                    "issues": issues
                },
                "stats": {
                    "retrieval_time_ms": retrieval_result.processing_time_ms,
                    "chunks_found": retrieval_result.total_chunks,
                    "retrieval_score": retrieval_result.retrieval_score,
                    "inference_time_ms": llm_response.inference_time_ms,
                    "tokens_generated": llm_response.completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "answer": "",
                "citations": [],
                "stats": {}
            }
    
    def summarize_topic(self, topic: str) -> Dict[str, Any]:
        """
        Generate a summary of a topic across all documents.
        
        Args:
            topic: Topic to summarize
            
        Returns:
            Dictionary with summary and metadata
        """
        if not self.documents_loaded:
            return {
                "success": False,
                "error": "No documents loaded. Please load documents first.",
                "summary": "",
                "sources": [],
                "stats": {}
            }
        
        try:
            logger.info(f"Generating summary for topic: '{topic}'")
            
            # Generate summary
            summary_result = self.summarizer.summarize_topic(topic)
            
            return {
                "success": True,
                "topic": summary_result.topic,
                "summary": summary_result.summary_text,
                "sources": summary_result.sources_used,
                "coverage_score": summary_result.coverage_score,
                "quality_score": summary_result.quality_score,
                "word_count": summary_result.word_count,
                "stats": {
                    "processing_time_ms": summary_result.processing_time_ms,
                    "retrieval_stats": summary_result.retrieval_stats,
                    "llm_stats": summary_result.llm_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "summary": "",
                "sources": [],
                "stats": {}
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "system": {
                "initialized": self.initialized,
                "documents_loaded": self.documents_loaded,
                "docs_folder": str(self.docs_folder),
                "embeddings_dir": str(self.embeddings_dir),
                "embedding_model": self.embedding_model,
                "using_mock_llm": self.use_mock_llm
            }
        }
        
        if self.document_loader:
            stats["document_loader"] = self.document_loader.get_processing_stats()
        
        if self.text_processor:
            stats["text_processor"] = self.text_processor.get_processing_stats()
        
        if self.vector_store:
            stats["vector_store"] = self.vector_store.get_stats()
        
        if self.retriever:
            stats["retriever"] = self.retriever.get_retrieval_stats()
        
        if self.llm_interface:
            stats["llm_interface"] = self.llm_interface.get_performance_stats()
        
        if self.summarizer:
            stats["summarizer"] = self.summarizer.get_summarization_stats()
        
        return stats
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all loaded documents with metadata.
        
        Returns:
            List of document information
        """
        if not self.document_loader:
            return []
        
        documents = []
        for filename, metadata in self.document_loader.processed_documents.items():
            documents.append({
                "filename": metadata.filename,
                "title": metadata.title,
                "subject": metadata.subject,
                "author": metadata.author,
                "page_count": metadata.page_count,
                "file_size_mb": round(metadata.file_size / (1024 * 1024), 2),
                "checksum": metadata.checksum[:16] + "..."  # Truncated for display
            })
        
        return documents
    
    def get_topic_suggestions(self) -> List[str]:
        """
        Get suggested topics based on loaded documents.
        
        Returns:
            List of suggested topics
        """
        if not self.summarizer:
            return []
        
        return self.summarizer.get_topic_suggestions()
    
    def search_documents(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks across all documents.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not self.vector_store:
            return []
        
        search_results = self.vector_store.search(query, k=max_results)
        
        return [
            {
                "chunk_id": result.chunk_id,
                "document_name": result.document_name,
                "page_number": result.page_number,
                "section_title": result.section_title,
                "similarity_score": result.similarity_score,
                "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
            }
            for result in search_results
        ]


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Klaro Academic Chatbot")
    parser.add_argument("--docs-folder", default="./klaro_docs/", help="Path to documents folder")
    parser.add_argument("--embeddings-dir", default="./embeddings/", help="Path to embeddings directory")
    parser.add_argument("--models-dir", default="./models/", help="Path to models directory")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name") #parser.add_argument("--embedding-model",  help="Embedding model name") #default="hkunlp/instructor-xl",
    parser.add_argument("--llm-model", help="Path to LLM model file")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM for testing")
    parser.add_argument("--load-docs", action="store_true", help="Load documents on startup")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize Klaro system
    klaro = KlaroSystem(
        docs_folder=args.docs_folder,
        embeddings_dir=args.embeddings_dir,
        models_dir=args.models_dir,
        embedding_model= "sentence-transformers/all-MiniLM-L6-v2", #args.embedding_model,
        llm_model_path=args.llm_model,
        use_mock_llm=args.mock_llm
    )
    
    # Initialize system
    if not klaro.initialize():
        logger.error("Failed to initialize Klaro system")
        sys.exit(1)
    
    # Load documents if requested
    if args.load_docs:
        if not klaro.load_documents():
            logger.error("Failed to load documents")
            sys.exit(1)
    
    # Start interactive mode if requested
    if args.interactive:
        interactive_mode(klaro)
    else:
        # Print system information
        stats = klaro.get_system_stats()
        print("Klaro Academic Chatbot System")
        print("=" * 40)
        print(f"Initialized: {stats['system']['initialized']}")
        print(f"Documents loaded: {stats['system']['documents_loaded']}")
        print(f"Documents folder: {stats['system']['docs_folder']}")
        print(f"Embedding model: {stats['system']['embedding_model']}")
        print(f"Using mock LLM: {stats['system']['using_mock_llm']}")
        
        if stats['system']['documents_loaded']:
            print(f"\nDocument Statistics:")
            print(f"- Total documents: {stats.get('document_loader', {}).get('total_documents', 0)}")
            print(f"- Total pages: {stats.get('document_loader', {}).get('total_pages', 0)}")
            print(f"- Total chunks: {stats.get('vector_store', {}).get('total_vectors', 0)}")
        
        print("\nUse --interactive flag to start interactive mode")
        print("Use --load-docs flag to load documents")


def interactive_mode(klaro: KlaroSystem):
    """Interactive command-line interface."""
    print("\nKlaro Academic Chatbot - Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  ask <question>     - Ask a question")
    print("  summarize <topic>  - Summarize a topic")
    print("  search <query>     - Search documents")
    print("  load               - Load documents")
    print("  docs               - List documents")
    print("  topics             - Get topic suggestions")
    print("  stats              - Show system statistics")
    print("  help               - Show this help")
    print("  quit               - Exit")
    print()
    
    while True:
        try:
            user_input = input("klaro> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(None, 1)
            command = parts[0].lower()
            
            if command == "quit":
                break
            elif command == "help":
                print("Available commands: ask, summarize, search, load, docs, topics, stats, help, quit")
            elif command == "load":
                print("Loading documents...")
                if klaro.load_documents():
                    print("Documents loaded successfully!")
                else:
                    print("Failed to load documents.")
            elif command == "ask":
                if len(parts) < 2:
                    print("Usage: ask <question>")
                    continue
                
                question = parts[1]
                result = klaro.ask_question(question)
                
                if result["success"]:
                    print(f"\nAnswer: {result['answer']}")
                    if result["citations"]:
                        print(f"\nSources:")
                        for i, citation in enumerate(result["citations"][:3], 1):
                            print(f"{i}. {citation['document_name']}, Page {citation['page_number']}")
                else:
                    print(f"Error: {result['error']}")
            
            elif command == "summarize":
                if len(parts) < 2:
                    print("Usage: summarize <topic>")
                    continue
                
                topic = parts[1]
                result = klaro.summarize_topic(topic)
                
                if result["success"]:
                    print(f"\nSummary of '{topic}':")
                    print(result["summary"])
                    print(f"\nQuality Score: {result['quality_score']:.2f}")
                    print(f"Coverage Score: {result['coverage_score']:.2f}")
                else:
                    print(f"Error: {result['error']}")
            
            elif command == "search":
                if len(parts) < 2:
                    print("Usage: search <query>")
                    continue
                
                query = parts[1]
                results = klaro.search_documents(query, max_results=5)
                
                if results:
                    print(f"\nSearch results for '{query}':")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['document_name']}, Page {result['page_number']}")
                        print(f"   Score: {result['similarity_score']:.3f}")
                        print(f"   Preview: {result['content_preview']}")
                        print()
                else:
                    print("No results found.")
            
            elif command == "docs":
                documents = klaro.list_documents()
                if documents:
                    print("\nLoaded Documents:")
                    for doc in documents:
                        print(f"- {doc['filename']} ({doc['page_count']} pages, {doc['file_size_mb']} MB)")
                        if doc['title']:
                            print(f"  Title: {doc['title']}")
                else:
                    print("No documents loaded.")
            
            elif command == "topics":
                topics = klaro.get_topic_suggestions()
                if topics:
                    print("\nSuggested Topics:")
                    for topic in topics[:10]:
                        print(f"- {topic}")
                else:
                    print("No topic suggestions available.")
            
            elif command == "stats":
                stats = klaro.get_system_stats()
                print("\nSystem Statistics:")
                for category, data in stats.items():
                    print(f"\n{category.title()}:")
                    for key, value in data.items():
                        print(f"  {key}: {value}")
            
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("Goodbye!")


if __name__ == "__main__":
    main()

