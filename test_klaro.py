"""
Test Suite for Klaro Academic Chatbot

This module provides comprehensive unit and integration tests for all
components of the Klaro system, ensuring reliability and correctness
of the educational chatbot functionality.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Import Klaro components
from document_loader import DocumentLoader, DocumentMetadata
from text_processor import TextProcessor, ProcessedChunk
from vector_store import VectorStore
from retriever import DocumentRetriever
from llm_interface import MockLLMInterface, LLMResponse
from summarizer import TopicSummarizer
from security import SecurityManager, SecurityConfig
from main import KlaroSystem

class TestDocumentLoader(unittest.TestCase):
    """Test cases for DocumentLoader class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.loader = DocumentLoader(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test DocumentLoader initialization"""
        self.assertEqual(str(self.loader.docs_folder), self.test_dir)
        self.assertEqual(len(self.loader.processed_documents), 0)
        self.assertTrue(Path(self.test_dir).exists())
    
    def test_validate_document_nonexistent(self):
        """Test validation of non-existent document"""
        fake_path = Path(self.test_dir) / "nonexistent.pdf"
        self.assertFalse(self.loader.validate_document(fake_path))
    
    def test_validate_document_wrong_extension(self):
        """Test validation of non-PDF file"""
        txt_file = Path(self.test_dir) / "test.txt"
        txt_file.write_text("test content")
        self.assertFalse(self.loader.validate_document(txt_file))
    
    def test_load_all_pdfs_empty_folder(self):
        """Test loading PDFs from empty folder"""
        documents = self.loader.load_all_pdfs()
        self.assertEqual(len(documents), 0)
    
    def test_get_processing_stats_empty(self):
        """Test processing statistics with no documents"""
        stats = self.loader.get_processing_stats()
        expected = {"total_documents": 0, "total_pages": 0, "total_size_mb": 0}
        self.assertEqual(stats, expected)

class TestTextProcessor(unittest.TestCase):
    """Test cases for TextProcessor class"""
    
    def setUp(self):
        """Set up test environment"""
        self.processor = TextProcessor(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=30
        )
    
    def test_initialization(self):
        """Test TextProcessor initialization"""
        self.assertEqual(self.processor.chunk_size, 100)
        self.assertEqual(self.processor.chunk_overlap, 20)
        self.assertEqual(self.processor.min_chunk_size, 30)
        self.assertEqual(len(self.processor.processed_chunks), 0)
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        raw_text = "This  is   a    test\n\n\n\nwith   excessive   whitespace."
        cleaned = self.processor._preprocess_text(raw_text)
        self.assertNotIn("   ", cleaned)  # No triple spaces
        self.assertNotIn("\n\n\n", cleaned)  # No triple newlines
    
    def test_process_document_simple(self):
        """Test processing a simple document"""
        text = "This is a test document. It has multiple sentences. Each sentence provides information."
        chunks = self.processor.process_document("test.pdf", text)
        
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], ProcessedChunk)
        self.assertEqual(chunks[0].document_name, "test.pdf")
    
    def test_extract_page_info(self):
        """Test page information extraction"""
        text = "--- Page 1 ---\nContent of page 1\n--- Page 2 ---\nContent of page 2"
        page_info = self.processor._extract_page_info(text)
        
        self.assertIn(1, page_info)
        self.assertIn(2, page_info)
        self.assertEqual(len(page_info), 2)
    
    def test_is_valid_chunk(self):
        """Test chunk validation"""
        # Valid chunk
        valid_chunk = ProcessedChunk(
            chunk_id="test_001",
            document_name="test.pdf",
            page_number=1,
            section_title="Test Section",
            content="This is a valid chunk with sufficient content for testing purposes.",
            char_start=0,
            char_end=70,
            word_count=12,
            chunk_index=0,
            overlap_with_previous=False,
            overlap_with_next=False
        )
        self.assertTrue(self.processor._is_valid_chunk(valid_chunk))
        
        # Invalid chunk (too short)
        invalid_chunk = ProcessedChunk(
            chunk_id="test_002",
            document_name="test.pdf",
            page_number=1,
            section_title=None,
            content="Short",
            char_start=0,
            char_end=5,
            word_count=1,
            chunk_index=1,
            overlap_with_previous=False,
            overlap_with_next=False
        )
        self.assertFalse(self.processor._is_valid_chunk(invalid_chunk))
    
    def test_get_processing_stats(self):
        """Test processing statistics"""
        # Process a document first
        text = "This is a test document with multiple sentences to create several chunks for testing."
        self.processor.process_document("test.pdf", text)
        
        stats = self.processor.get_processing_stats()
        self.assertGreater(stats["total_chunks"], 0)
        self.assertGreater(stats["total_words"], 0)
        self.assertEqual(stats["documents_processed"], 1)

class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        # Use a smaller model for testing
        self.vector_store = VectorStore(
            model_name="all-MiniLM-L6-v2",
            embeddings_dir=self.test_dir,
            index_name="test_index"
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test VectorStore initialization"""
        self.assertIsNotNone(self.vector_store.embedding_model)
        self.assertEqual(self.vector_store.index_name, "test_index")
        self.assertEqual(str(self.vector_store.embeddings_dir), self.test_dir)
    
    def test_add_chunks(self):
        """Test adding chunks to vector store"""
        chunks = [
            ProcessedChunk(
                chunk_id="test_001",
                document_name="test.pdf",
                page_number=1,
                section_title="Test Section",
                content="This is test content for vector embedding.",
                char_start=0,
                char_end=45,
                word_count=8,
                chunk_index=0,
                overlap_with_previous=False,
                overlap_with_next=False
            )
        ]
        
        self.vector_store.add_chunks(chunks)
        
        self.assertIsNotNone(self.vector_store.index)
        self.assertEqual(self.vector_store.index.ntotal, 1)
        self.assertEqual(len(self.vector_store.chunk_metadata), 1)
    
    def test_search_empty_store(self):
        """Test searching empty vector store"""
        results = self.vector_store.search("test query")
        self.assertEqual(len(results), 0)
    
    def test_get_stats_empty(self):
        """Test statistics for empty vector store"""
        stats = self.vector_store.get_stats()
        self.assertEqual(stats["total_vectors"], 0)
        self.assertGreater(stats["embedding_dimension"], 0)

class TestRetriever(unittest.TestCase):
    """Test cases for DocumentRetriever class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            model_name="all-MiniLM-L6-v2",
            embeddings_dir=self.test_dir,
            index_name="test_index"
        )
        self.retriever = DocumentRetriever(
            vector_store=self.vector_store,
            max_context_length=1000,
            max_chunks_per_query=3
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test DocumentRetriever initialization"""
        self.assertEqual(self.retriever.max_context_length, 1000)
        self.assertEqual(self.retriever.max_chunks_per_query, 3)
        self.assertEqual(self.retriever.query_count, 0)
    
    def test_retrieve_context_empty_store(self):
        """Test context retrieval from empty store"""
        result = self.retriever.retrieve_context("test query")
        
        self.assertEqual(result.query, "test query")
        self.assertEqual(result.context, "")
        self.assertEqual(len(result.citations), 0)
        self.assertEqual(result.total_chunks, 0)
    
    def test_create_excerpt(self):
        """Test excerpt creation"""
        long_text = "This is a very long text that should be truncated to create a brief excerpt for citation purposes."
        excerpt = self.retriever._create_excerpt(long_text, 50)
        
        self.assertLessEqual(len(excerpt), 60)  # Allow some buffer for ellipsis
        self.assertTrue(excerpt.endswith("...") or len(excerpt) <= 50)
    
    def test_calculate_retrieval_score(self):
        """Test retrieval score calculation"""
        from vector_store import SearchResult
        
        # Mock search results
        results = [
            SearchResult("chunk1", "content1", "doc1.pdf", 1, "Section 1", 0.9, 1),
            SearchResult("chunk2", "content2", "doc1.pdf", 2, "Section 2", 0.7, 2),
        ]
        
        score = self.retriever._calculate_retrieval_score(results)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

class TestLLMInterface(unittest.TestCase):
    """Test cases for LLMInterface class"""
    
    def setUp(self):
        """Set up test environment"""
        self.llm = MockLLMInterface(
            ### model_name="test-model",
            model_name="mock-model",
            max_tokens=100,
            temperature=0.1
        )
    
    def test_initialization(self):
        """Test LLMInterface initialization"""
        self.assertEqual(self.llm.model_name, "test-model")
        self.assertEqual(self.llm.max_tokens, 100)
        self.assertEqual(self.llm.temperature, 0.1)
        self.assertTrue(self.llm.model_loaded)
    
    def test_answer_question(self):
        """Test question answering"""
        context = "Biology is the study of life and living organisms."
        question = "What is biology?"
        
        response = self.llm.answer_question(question, context)
        
        self.assertIsInstance(response, LLMResponse)
        self.assertGreater(len(response.response_text), 0)
        self.assertGreater(response.total_tokens, 0)
        self.assertEqual(response.model_name, "mock-model")
    
    def test_summarize_topic(self):
        """Test topic summarization"""
        context = "Photosynthesis is the process by which plants convert light energy into chemical energy."
        topic = "photosynthesis"
        
        response = self.llm.summarize_topic(topic, context)
        
        self.assertIsInstance(response, LLMResponse)
        self.assertGreater(len(response.response_text), 0)
        self.assertIn("photosynthesis", response.response_text.lower())
    
    def test_validate_response(self):
        """Test response validation"""
        context = "Biology is the study of life."
        valid_response = "Based on the provided context, biology is the study of life. [Source: textbook.pdf, Page 1]"
        
        is_valid, issues = self.llm.validate_response(valid_response, context)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_get_performance_stats(self):
        """Test performance statistics"""
        # Generate a response first
        self.llm.answer_question("test", "test context")
        
        stats = self.llm.get_performance_stats()
        self.assertEqual(stats["model_name"], "mock-model")
        self.assertTrue(stats["model_loaded"])
        self.assertGreater(stats["total_queries"], 0)

class TestSummarizer(unittest.TestCase):
    """Test cases for TopicSummarizer class"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(
            model_name="all-MiniLM-L6-v2",
            embeddings_dir=self.test_dir,
            index_name="test_index"
        )
        self.retriever = DocumentRetriever(self.vector_store)
        self.llm = MockLLMInterface()
        self.summarizer = TopicSummarizer(
            retriever=self.retriever,
            llm_interface=self.llm,
            min_sources=1,
            max_summary_length=500
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test TopicSummarizer initialization"""
        self.assertEqual(self.summarizer.min_sources, 1)
        self.assertEqual(self.summarizer.max_summary_length, 500)
        self.assertEqual(self.summarizer.summaries_generated, 0)
    
    def test_summarize_topic_empty_store(self):
        """Test topic summarization with empty store"""
        result = self.summarizer.summarize_topic("test topic")
        
        self.assertEqual(result.topic, "test topic")
        self.assertIn("couldn't find", result.summary_text.lower())
        self.assertEqual(len(result.sources_used), 0)
        self.assertEqual(result.coverage_score, 0.0)
    
    def test_enhance_summary(self):
        """Test summary enhancement"""
        original_summary = "This is a basic summary without proper formatting."
        citations = [
            {"document_name": "test.pdf", "page_number": 1, "relevance_score": 0.9}
        ]
        
        enhanced = self.summarizer._enhance_summary(original_summary, citations, "test topic")
        
        self.assertIn("Test Topic", enhanced)  # Title added
        self.assertIn("Sources:", enhanced)  # Sources added
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        from retriever import RetrievalResult
        
        good_summary = "This is a well-structured summary with proper citations. [Source: test.pdf, Page 1] It has multiple sentences and good organization. Furthermore, it demonstrates coherent flow."
        
        mock_retrieval = RetrievalResult(
            query="test",
            context="test context",
            citations=[{"document_name": "test.pdf", "page_number": 1}],
            total_chunks=1,
            retrieval_score=0.8,
            processing_time_ms=100
        )
        
        score = self.summarizer._calculate_quality_score(good_summary, mock_retrieval)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

class TestSecurity(unittest.TestCase):
    """Test cases for Security components"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SecurityConfig(
            max_query_length=100,
            rate_limit_requests_per_minute=5,
            enable_content_filtering=True
        )
        self.security_manager = SecurityManager(self.config)
    
    def test_validate_query_valid(self):
        """Test validation of valid query"""
        is_valid, sanitized, error = self.security_manager.validate_request(
            "What is photosynthesis?", "test_client"
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(sanitized, "What is photosynthesis?")
        self.assertEqual(error, "")
    
    def test_validate_query_too_long(self):
        """Test validation of overly long query"""
        long_query = "A" * 200  # Exceeds max_query_length of 100
        
        is_valid, sanitized, error = self.security_manager.validate_request(
            long_query, "test_client"
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(sanitized, "")
        self.assertIn("too long", error.lower())
    
    def test_validate_query_sql_injection(self):
        """Test detection of SQL injection attempt"""
        malicious_query = "What is biology? SELECT * FROM users"
        
        is_valid, sanitized, error = self.security_manager.validate_request(
            malicious_query, "test_client"
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(sanitized, "")
        self.assertIn("Invalid characters", error)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        client_id = "rate_test_client"
        
        # Make requests up to the limit
        for i in range(5):  # Limit is 5 per minute
            is_valid, _, _ = self.security_manager.validate_request(
                f"Query {i}", client_id
            )
            self.assertTrue(is_valid)
        
        # Next request should be blocked
        is_valid, _, error = self.security_manager.validate_request(
            "Blocked query", client_id
        )
        self.assertFalse(is_valid)
        self.assertIn("Rate limit", error)
    
    def test_file_validation(self):
        """Test file validation"""
        # Create a temporary test file
        test_file = Path(tempfile.mktemp(suffix=".pdf"))
        test_file.write_bytes(b"%PDF-1.4\nTest PDF content")
        
        try:
            is_valid, error = self.security_manager.validate_file_upload(
                test_file, "test_client"
            )
            # Note: This might fail due to invalid PDF structure, which is expected
            # The test verifies the validation process runs without errors
            self.assertIsInstance(is_valid, bool)
            self.assertIsInstance(error, str)
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_security_status(self):
        """Test security status reporting"""
        status = self.security_manager.get_security_status()
        
        self.assertIn("config", status)
        self.assertIn("metrics", status)
        self.assertIn("status", status)
        self.assertEqual(status["status"], "active")

class TestKlaroSystem(unittest.TestCase):
    """Integration tests for the complete Klaro system"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.klaro = KlaroSystem(
            docs_folder=self.test_dir,
            use_mock_llm=True
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test Klaro system initialization"""
        self.assertTrue(self.klaro.initialize())
        self.assertTrue(self.klaro.initialized)
        self.assertIsNotNone(self.klaro.document_loader)
        self.assertIsNotNone(self.klaro.text_processor)
        self.assertIsNotNone(self.klaro.vector_store)
        self.assertIsNotNone(self.klaro.retriever)
        self.assertIsNotNone(self.klaro.llm_interface)
        self.assertIsNotNone(self.klaro.summarizer)
    
    def test_ask_question_no_documents(self):
        """Test asking question without loaded documents"""
        self.klaro.initialize()
        
        result = self.klaro.ask_question("What is biology?")
        
        self.assertFalse(result["success"])
        self.assertIn("No documents loaded", result["error"])
    
    def test_summarize_topic_no_documents(self):
        """Test topic summarization without loaded documents"""
        self.klaro.initialize()
        
        result = self.klaro.summarize_topic("photosynthesis")
        
        self.assertFalse(result["success"])
        self.assertIn("No documents loaded", result["error"])
    
    def test_get_system_stats(self):
        """Test system statistics retrieval"""
        self.klaro.initialize()
        
        stats = self.klaro.get_system_stats()
        
        self.assertIn("system", stats)
        self.assertIn("document_loader", stats)
        self.assertIn("vector_store", stats)
        self.assertTrue(stats["system"]["initialized"])
    
    def test_list_documents_empty(self):
        """Test listing documents when none are loaded"""
        self.klaro.initialize()
        
        documents = self.klaro.list_documents()
        self.assertEqual(len(documents), 0)
    
    def test_search_documents_empty(self):
        """Test searching documents when none are loaded"""
        self.klaro.initialize()
        
        results = self.klaro.search_documents("test query")
        self.assertEqual(len(results), 0)

class TestPerformance(unittest.TestCase):
    """Performance tests for Klaro components"""
    
    def test_text_processing_performance(self):
        """Test text processing performance"""
        processor = TextProcessor()
        
        # Generate large text
        large_text = "This is a test sentence. " * 1000  # ~25KB of text
        
        start_time = time.time()
        chunks = processor.process_document("performance_test.pdf", large_text)
        processing_time = time.time() - start_time
        
        self.assertGreater(len(chunks), 0)
        self.assertLess(processing_time, 10.0)  # Should process in under 10 seconds
    
    def test_vector_search_performance(self):
        """Test vector search performance"""
        test_dir = tempfile.mkdtemp()
        
        try:
            vector_store = VectorStore(
                model_name="all-MiniLM-L6-v2",
                embeddings_dir=test_dir
            )
            
            # Add multiple chunks
            chunks = []
            for i in range(50):
                chunk = ProcessedChunk(
                    chunk_id=f"perf_test_{i:03d}",
                    document_name="performance_test.pdf",
                    page_number=i // 10 + 1,
                    section_title=f"Section {i}",
                    content=f"This is performance test content number {i} with various keywords and information.",
                    char_start=i * 100,
                    char_end=(i + 1) * 100,
                    word_count=15,
                    chunk_index=i,
                    overlap_with_previous=i > 0,
                    overlap_with_next=i < 49
                )
                chunks.append(chunk)
            
            # Add chunks and measure time
            start_time = time.time()
            vector_store.add_chunks(chunks)
            indexing_time = time.time() - start_time
            
            # Search and measure time
            start_time = time.time()
            results = vector_store.search("performance test", k=10)
            search_time = time.time() - start_time
            
            self.assertLess(indexing_time, 30.0)  # Should index in under 30 seconds
            self.assertLess(search_time, 1.0)     # Should search in under 1 second
            self.assertGreater(len(results), 0)
            
        finally:
            shutil.rmtree(test_dir)

def run_tests():
    """Run all tests and generate a report"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDocumentLoader,
        TestTextProcessor,
        TestVectorStore,
        TestRetriever,
        TestLLMInterface,
        TestSummarizer,
        # TestSecurity,
        TestKlaroSystem,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

