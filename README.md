# Klaro Academic Chatbot

**Intelligent Curriculum-Based Learning Assistant**

Klaro is a local-first, open-source AI chatbot designed to improve learning outcomes by providing accurate, citation-backed answers and summaries based exclusively on your textbook content. Built with privacy and educational integrity in mind, Klaro operates completely offline using only open-source models and tools.

## 🎯 Key Features

### 📖 Contextual Q&A from Textbooks
- Answers generated exclusively from preloaded textbooks (no external knowledge)
- Uses retrieval-augmented generation (RAG) with local vector search
- Supports ~20 PDFs, each up to 500 pages
- No hallucinations or unsupported responses

### 🔗 Hyperlinked References
- Every response includes citation of exact source (filename, page/section)
- Example format: "See Grade8_Science.pdf - Page 115, Section 3.2"
- Clickable citations for easy verification

### 📚 Topic Summarization Across Documents
- Can summarize concepts (e.g., "photosynthesis") using content from multiple books/chapters
- Produces concise, source-cited overviews
- Intelligent content synthesis from diverse sources

### 🔐 Privacy & Security
- Complete offline operation with no external API calls
- Local-first architecture with no data transmission
- Input validation and rate limiting
- Secure file handling and processing

### 🎨 User-Friendly Interface
- Clean, intuitive Gradio web interface
- Real-time chat functionality
- Document management and statistics
- Search and exploration tools

## 🏗️ Architecture

Klaro follows a modular RAG (Retrieval-Augmented Generation) architecture:

```
[PDF Documents] → [Text Extraction] → [Chunking] → [Embedding Generation] → [Vector Store]
                                                                                    ↓
[User Query] → [Query Embedding] → [Similarity Search] → [Context Assembly] → [LLM Inference] → [Response + Citations]
```

### Core Components

- **Document Loader**: Robust PDF processing with PyMuPDF
- **Text Processor**: Intelligent chunking with context preservation
- **Vector Store**: High-performance similarity search with FAISS
- **Retriever**: Context assembly and citation generation
- **LLM Interface**: Local language model integration
- **Summarizer**: Multi-document topic synthesis
- **Security Manager**: Input validation and system protection

## 🛠️ Technical Stack

| Component | Tool/Library | Purpose |
|-----------|--------------|---------|
| Language Model | Mistral-7B, Phi-3, or OpenChat via llama-cpp | Fast, local, high-accuracy LLMs |
| Embeddings | Instructor-XL or bge-large-en | Best-in-class open-source sentence embeddings |
| Vector Store | FAISS | Local, fast, high-accuracy vector search |
| PDF Parsing | PyMuPDF (fitz) | Fast, accurate layout-preserving PDF parsing |
| Text Splitting | LangChain | Reliable recursive chunking strategy |
| RAG Framework | LangChain | Modular, mature RAG pipeline support |
| UI Framework | Gradio | Simple, effective web interface |
| LLM Runtime | llama-cpp-python | Lightweight local inference for GGUF models |

## 📁 Project Structure

```
klaro/
├── klaro_docs/          # Textbook PDFs (input)
├── models/              # GGUF LLM models (Mistral, Phi)
├── embeddings/          # Vector store files
├── main.py              # Entry point for chat & search
├── document_loader.py   # PDF processing and metadata extraction
├── text_processor.py    # Text chunking and preprocessing
├── vector_store.py      # Embedding generation and similarity search
├── retriever.py         # Search and context handler
├── llm_interface.py     # Local LLM integration
├── summarizer.py        # Topic summarization logic
├── security.py          # Input validation and security
├── ui.py                # Gradio web interface
├── test_klaro.py        # Comprehensive test suite
├── requirements.txt     # Python dependencies
├── architecture.md      # Detailed system architecture
└── README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended for larger models)
- 10GB+ free disk space

### Installation

1. **Clone or download the project**
   ```bash
   # If using git
   git clone <repository-url>
   cd klaro
   
   # Or extract from zip file
   unzip klaro.zip
   cd klaro
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your textbooks**
   ```bash
   # Place PDF files in the klaro_docs folder
   cp your_textbooks/*.pdf klaro_docs/
   ```

4. **Download a language model (optional)**
   ```bash
   # Create models directory
   mkdir -p models
   
   # Download a GGUF model (example)
   # Visit https://huggingface.co/models?search=gguf for options
   # Example: Mistral-7B-Instruct GGUF
   wget -O models/mistral-7b-instruct.gguf [MODEL_URL]
   ```

### Running Klaro

#### Option 1: Web Interface (Recommended)
```bash
# Start the web interface
python ui.py

# Or with custom settings
python ui.py --port 8080 --mock-llm
```

Then open your browser to `http://localhost:7860`

#### Option 2: Command Line Interface
```bash
# Interactive mode
python main.py --interactive --load-docs --mock-llm

# Or with real LLM
python main.py --interactive --load-docs --llm-model models/mistral-7b-instruct.gguf
```

#### Option 3: Python API
```python
from main import KlaroSystem

# Initialize system
klaro = KlaroSystem(use_mock_llm=True)  # Set False for real LLM
klaro.initialize()
klaro.load_documents()

# Ask questions
result = klaro.ask_question("What is photosynthesis?")
print(result["answer"])

# Generate summaries
summary = klaro.summarize_topic("cell structure")
print(summary["summary"])
```

## 📖 Usage Guide

### Loading Documents

1. Place PDF textbooks in the `klaro_docs/` folder
2. Click "Load Documents" in the web interface
3. Wait for processing to complete
4. Check the "System Information" tab for loading statistics

### Asking Questions

1. Navigate to the "Ask Questions" tab
2. Type your question in natural language
3. Click "Ask" or press Enter
4. Review the answer with citations
5. Click on citations to verify sources

**Example Questions:**
- "What is photosynthesis?"
- "Explain the structure of a cell"
- "How does the digestive system work?"
- "What are the main types of chemical bonds?"

### Generating Summaries

1. Go to the "Summarize Topics" tab
2. Enter a topic (e.g., "photosynthesis", "cell division")
3. Click "Generate Summary"
4. Review the comprehensive summary with sources

### Searching Documents

1. Use the "Search Documents" tab
2. Enter keywords or phrases
3. Browse relevant chunks across all documents
4. Use results to inform your questions or summaries

## ⚙️ Configuration

### Security Settings

Edit `security.py` to customize:
- Maximum query length
- Rate limiting parameters
- File upload restrictions
- Content filtering rules

### Performance Tuning

Adjust settings in component initialization:
- Chunk size and overlap (text processing)
- Vector search parameters (retrieval)
- LLM context window and temperature
- Embedding model selection

### Model Selection

Klaro supports various local LLM models:
- **Mistral-7B-Instruct**: Balanced performance and accuracy
- **Phi-3-Mini**: Lightweight, fast inference
- **OpenChat**: Optimized for conversational tasks

Download GGUF format models from Hugging Face for best performance.

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_klaro.py

# Run specific test categories
python -m unittest test_klaro.TestDocumentLoader
python -m unittest test_klaro.TestSecurity
python -m unittest test_klaro.TestPerformance
```

## 📊 Performance Metrics

### Evaluation Targets

| Metric | Target | Method |
|--------|--------|--------|
| Response Accuracy | ≥ 95% | Human review vs. source |
| Citation Correctness | ≥ 98% | Link resolves to correct section |
| Summarization Coherence | ≥ 4/5 | Manual scoring by reviewers |
| Latency | ≤ 3 sec | Time per query (on CPU/GPU) |
| Hallucination Rate | ≤ 1% | Unsubstantiated output detections |

### System Requirements

- **Minimum**: 8GB RAM, 4-core CPU, 10GB storage
- **Recommended**: 16GB RAM, 8-core CPU, 20GB storage
- **Optimal**: 32GB RAM, GPU support, SSD storage

## 🔒 Security Features

### Input Validation
- SQL injection prevention
- XSS attack protection
- Content filtering for educational appropriateness
- File type and size validation

### Rate Limiting
- Per-client request limits
- Sliding window algorithm
- Automatic cleanup of old entries
- Configurable thresholds

### Privacy Protection
- No external API calls
- Local data processing only
- Secure file handling
- No user data transmission

## 🐛 Troubleshooting

### Common Issues

**"No documents loaded" error**
- Ensure PDF files are in `klaro_docs/` folder
- Check file permissions and formats
- Verify PDFs are not corrupted

**Slow performance**
- Reduce chunk size in text processor
- Use smaller embedding models
- Enable GPU acceleration if available
- Increase system RAM

**Model loading errors**
- Verify model file path and format (GGUF)
- Check available system memory
- Try smaller models for testing
- Use mock LLM for development

**Empty search results**
- Ensure documents are properly loaded
- Try different keywords or phrases
- Check document content quality
- Verify embedding model is working

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python test_klaro.py

# Format code
black *.py

# Check style
flake8 *.py
```

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **LangChain**: RAG pipeline framework
- **Sentence Transformers**: High-quality embeddings
- **FAISS**: Efficient similarity search
- **PyMuPDF**: Robust PDF processing
- **Gradio**: User-friendly web interfaces
- **llama.cpp**: Optimized local LLM inference

## 📞 Support

For questions, issues, or contributions:

1. Check the troubleshooting section above
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Join our community discussions

---

**Klaro Academic Chatbot** - Empowering education through intelligent, privacy-focused AI assistance.

