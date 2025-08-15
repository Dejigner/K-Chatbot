"""
User Interface Module for Klaro Academic Chatbot

This module provides a Gradio-based web interface for the Klaro system,
offering an intuitive way to interact with the educational chatbot for
Q&A, summarization, and document management.
"""

import argparse
import gradio as gr
import logging
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd

from main import KlaroSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KlaroUI:
    """
    Gradio-based user interface for Klaro Academic Chatbot.
    
    Features:
    - Interactive chat interface for Q&A
    - Topic summarization interface
    - Document management and upload
    - System statistics and monitoring
    - Search functionality
    """
    
    def __init__(self, 
                 klaro_system: Optional[KlaroSystem] = None,
                 docs_folder: str = "./klaro_docs/",
                 llm_model_path: Optional[str] = None,  # before: (param did not exist)
                 use_mock_llm: bool = False):           # before: use_mock_llm: bool = True
        """
        Initialize the Klaro UI.
        
        Args:
            klaro_system: Pre-initialized Klaro system (optional)
            docs_folder: Path to documents folder
            llm_model_path: Path/name of local LLM model (disables mock)
            use_mock_llm: Whether to use mock LLM for demo purposes
        """
        self.docs_folder = Path(docs_folder)
        self.docs_folder.mkdir(parents=True, exist_ok=True)

        self.klaro = klaro_system or KlaroSystem(
            docs_folder=str(self.docs_folder),
            llm_model_path=llm_model_path,            # before: not passed
            use_mock_llm=use_mock_llm,                # before: use_mock_llm=use_mock_llm (defaulted True)
        )
        if not self.klaro.initialized:
            self.klaro.initialize()

        # UI state
        self.chat_history = []
        self.system_initialized = False
        
        logger.info("Klaro UI initialized")
    
    def _save_uploaded(self, files: Optional[List[str]]) -> List[Path]:
        """Save uploaded files into the docs folder, avoiding overwriting."""
        saved: List[Path] = []
        if not files:
            return saved

        for f in files:
            src = Path(f)
            if not src.exists():
                continue

            dest = self.docs_folder / src.name
            # If same name exists and same size, skip as duplicate
            if dest.exists() and src.stat().st_size == dest.stat().st_size:
                logger.info(f"Skipping duplicate: {dest.name}")
                continue

            # If name exists but differs, make a unique name
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                i = 1
                while True:
                    alt = self.docs_folder / f"{stem} ({i}){suffix}"
                    if not alt.exists():
                        dest = alt
                        break
                    i += 1

            shutil.copy2(src, dest)
            saved.append(dest)
            logger.info(f"Saved upload: {dest}")
        return saved

    def upload_files_ui(self, files: Optional[List[str]]) -> Tuple[str, str]:
        """Handler for the file upload: save uploads."""
        saved = self._save_uploaded(files)
        names = ", ".join(p.name for p in saved) if saved else "none"
        return (
            f"Uploaded {len(saved)} file(s) to {self.docs_folder}.",
            f"Files: {names}\nClick 'Load Documents' or '🆕 Load New Documents' to index."
        )

    def load_documents_ui(self, files: Optional[List[str]]) -> Tuple[str, str]:
        """Handler for the Load Documents button: save uploads, then index."""
        saved = self._save_uploaded(files)
        ok = self.klaro.load_documents(force_reload=True)
        if ok: self.system_initialized = True  # before: (not updating system_initialized)
        status = f"Saved {len(saved)} file(s) to {self.docs_folder}."
        if ok:
            stats = self.klaro.get_system_stats()
            total_docs = stats.get("document_loader", {}).get("total_documents", 0)
            total_pages = stats.get("document_loader", {}).get("total_pages", 0)
            vectors = stats.get("vector_store", {}).get("total_vectors", 0)
            details = f"Indexed documents: {total_docs} | pages: {total_pages} | vectors: {vectors}"
        else:
            details = "Failed to load/index documents. Check logs for details."
        return status, details
    
    def load_new_documents_ui(self, files: Optional[List[str]]) -> Tuple[str, str]:  # before: (method did not exist)
        """Save uploads, then index only unprocessed/new docs."""
        saved = self._save_uploaded(files)
        ok = self.klaro.load_documents(force_reload=False)  # before: N/A
        status = f"Saved {len(saved)} file(s) to {self.docs_folder}."
        if ok:
            self.system_initialized = True  # before: N/A
            stats = self.klaro.get_system_stats()
            total_docs = stats.get("document_loader", {}).get("total_documents", 0)
            total_pages = stats.get("document_loader", {}).get("total_pages", 0)
            vectors = stats.get("vector_store", {}).get("total_vectors", 0)
            details = f"Indexed NEW documents only: {total_docs} | pages: {total_pages} | vectors: {vectors}"
        else:
            details = "Failed to load/index documents. Check logs for details."
        return status, details
    
    def ask_question_ui(self, question: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """
        Process a question through the UI.
        
        Args:
            question: User question
            history: Chat history
            
        Returns:
            Tuple of (updated_history, empty_input)
        """
        if not question.strip():
            return history, ""
        
        if not self.system_initialized:
            history.append([question, "⚠️ Please load documents first before asking questions."])
            return history, ""
        
        try:
            # Add user question to history
            history.append([question, "🤔 Thinking..."])
            
            # Fallback: if using mock LLM, compose an extractive answer from search results
            if getattr(self.klaro, "use_mock_llm", False):  # before: no fallback path
                results = self.klaro.search_documents(question, max_results=5)
                if results:
                    snippets = []
                    citations = []
                    for i, r in enumerate(results, 1):
                        txt = (r.get("content_preview") or r.get("content") or "").strip()
                        if txt:
                            snippets.append(f"- {txt[:400]}")
                        name = r.get("document_name", "Unknown")
                        page = r.get("page_number", "?")
                        sect = r.get("section_title")
                        citations.append(f"{i}. {name}, Page {page}" + (f" ({sect})" if sect else ""))
                    response = "Here’s what the documents say:\n\n" + "\n".join(snippets)
                    if citations:
                        response += "\n\nSources:\n" + "\n".join(citations)
                    history[-1][1] = response
                    return history, ""
                # if no results, fall through to normal path
            # ...existing code...
            result = self.klaro.ask_question(question)
            
            if result["success"]:
                # Format the response with citations
                response = result["answer"]
                
                if result["citations"]:
                    response += "\n\n**Sources:**\n"
                    for i, citation in enumerate(result["citations"][:5], 1):
                        response += f"{i}. {citation['document_name']}, Page {citation['page_number']}"
                        if citation.get('section_title'):
                            response += f" ({citation['section_title']})"
                        response += f" - Relevance: {citation['relevance_score']:.2f}\n"
                
                # Add performance stats
                stats = result["stats"]
                response += f"\n*Retrieved {stats['chunks_found']} chunks in {stats['retrieval_time_ms']}ms*"
                
                # Update the last message in history
                history[-1][1] = response
            else:
                history[-1][1] = f"❌ Error: {result['error']}"
            
            return history, ""
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            history[-1][1] = f"❌ Error: {str(e)}"
            return history, ""
    
    def summarize_topic_ui(self, topic: str) -> str:
        """
        Generate a topic summary through the UI.
        
        Args:
            topic: Topic to summarize
            
        Returns:
            Summary text with metadata
        """
        if not topic.strip():
            return "Please enter a topic to summarize."
        
        if not self.system_initialized:
            return "⚠️ Please load documents first before requesting summaries."
        
        try:
            logger.info(f"Generating summary for: {topic}")
            
            result = self.klaro.summarize_topic(topic)
            
            if result["success"]:
                summary = f"# Summary: {result['topic']}\n\n"
                summary += result["summary"]
                
                # Add metadata
                summary += f"\n\n---\n"
                summary += f"**Quality Score:** {result['quality_score']:.2f}/1.0\n"
                summary += f"**Coverage Score:** {result['coverage_score']:.2f}/1.0\n"
                summary += f"**Word Count:** {result['word_count']}\n"
                summary += f"**Processing Time:** {result['stats']['processing_time_ms']}ms\n"
                
                # Add sources
                if result["sources"]:
                    summary += f"\n**Sources Used:** {len(result['sources'])} documents\n"
                    for source in result["sources"][:5]:
                        summary += f"- {source['document_name']}, Page {source['page_number']}\n"
                
                return summary
            else:
                return f"❌ Error: {result['error']}"
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def search_documents_ui(self, query: str) -> str:
        """
        Search documents through the UI.
        
        Args:
            query: Search query
            
        Returns:
            Formatted search results
        """
        if not query.strip():
            return "Please enter a search query."
        
        if not self.system_initialized:
            return "⚠️ Please load documents first before searching."
        
        try:
            results = self.klaro.search_documents(query, max_results=10)
            
            if results:
                output = f"# Search Results for '{query}'\n\n"
                output += f"Found {len(results)} relevant chunks:\n\n"
                
                for i, result in enumerate(results, 1):
                    output += f"## {i}. {result['document_name']} (Page {result['page_number']})\n"
                    if result['section_title']:
                        output += f"**Section:** {result['section_title']}\n"
                    output += f"**Relevance:** {result['similarity_score']:.3f}\n\n"
                    output += f"{result['content_preview']}\n\n"
                    output += "---\n\n"
                
                return output
            else:
                return f"No results found for '{query}'. Try different keywords or check if documents are loaded."
                
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def get_system_stats_ui(self) -> str:
        """
        Get system statistics for the UI.
        
        Returns:
            Formatted system statistics
        """
        try:
            stats = self.klaro.get_system_stats()
            
            output = "# Klaro System Statistics\n\n"
            
            # System overview
            system_stats = stats.get("system", {})
            output += "## System Overview\n"
            output += f"- **Status:** {'✅ Ready' if system_stats.get('documents_loaded') else '⚠️ Documents not loaded'}\n"
            output += f"- **Documents Folder:** {system_stats.get('docs_folder', 'N/A')}\n"
            output += f"- **Embedding Model:** {system_stats.get('embedding_model', 'N/A')}\n"
            output += f"- **Using Mock LLM:** {'Yes' if system_stats.get('using_mock_llm') else 'No'}\n"  # ...existing code...
            output += f"- **LLM Model:** {stats.get('llm_interface', {}).get('model_name', 'N/A')}\n\n"  # before: (line did not exist)
            
            # Document statistics
            doc_stats = stats.get("document_loader", {})
            if doc_stats:
                output += "## Document Statistics\n"
                output += f"- **Total Documents:** {doc_stats.get('total_documents', 0)}\n"
                output += f"- **Total Pages:** {doc_stats.get('total_pages', 0)}\n"
                output += f"- **Total Size:** {doc_stats.get('total_size_mb', 0)} MB\n"
                output += f"- **Average Pages per Document:** {doc_stats.get('average_pages_per_doc', 0)}\n\n"
            
            # Vector store statistics
            vector_stats = stats.get("vector_store", {})
            if vector_stats:
                output += "## Vector Store Statistics\n"
                output += f"- **Total Vectors:** {vector_stats.get('total_vectors', 0)}\n"
                output += f"- **Embedding Dimension:** {vector_stats.get('embedding_dimension', 0)}\n"
                output += f"- **Index Size:** {vector_stats.get('index_size_mb', 0)} MB\n"
                output += f"- **Documents Indexed:** {vector_stats.get('total_documents', 0)}\n\n"
            
            # Performance statistics
            retriever_stats = stats.get("retriever", {})
            llm_stats = stats.get("llm_interface", {})
            
            if retriever_stats or llm_stats:
                output += "## Performance Statistics\n"
                
                if retriever_stats:
                    output += f"- **Total Queries:** {retriever_stats.get('total_queries', 0)}\n"
                    output += f"- **Average Retrieval Time:** {retriever_stats.get('average_processing_time_ms', 0):.1f}ms\n"
                
                if llm_stats:
                    output += f"- **LLM Queries:** {llm_stats.get('total_queries', 0)}\n"
                    output += f"- **Average Inference Time:** {llm_stats.get('average_inference_time_ms', 0):.1f}ms\n"
                    output += f"- **Total Tokens Generated:** {llm_stats.get('total_tokens_generated', 0)}\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def get_topic_suggestions_ui(self) -> str:
        """
        Get topic suggestions for the UI.
        
        Returns:
            Formatted topic suggestions
        """
        if not self.system_initialized:
            return "⚠️ Please load documents first to get topic suggestions."
        
        try:
            topics = self.klaro.get_topic_suggestions()
            
            if topics:
                output = "# Suggested Topics\n\n"
                output += "Based on your loaded documents, here are some topics you can explore:\n\n"
                
                for i, topic in enumerate(topics[:15], 1):
                    output += f"{i}. {topic}\n"
                
                output += "\n*Click on any topic above to use it in the summarization tab.*"
                return output
            else:
                return "No topic suggestions available. This may happen if documents don't have clear section headings."
                
        except Exception as e:
            logger.error(f"Error getting topic suggestions: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f5f5f5;
            margin-right: 20%;
        }
        .stats-container {
            background-color: #fafafa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        """
        
        with gr.Blocks(
            title="Klaro Academic Chatbot",
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🎓 Klaro Academic Chatbot
            
            **Intelligent Curriculum-Based Learning Assistant**
            
            Klaro helps you explore and understand your textbook content through AI-powered Q&A and summarization.
            All responses are grounded in your uploaded documents with proper citations.
            """)
            
            # System status
            with gr.Row():
                with gr.Column(scale=2):
                    system_status = gr.Markdown("📋 **Status:** Ready to load documents")
                with gr.Column(scale=1):
                    load_docs_btn = gr.Button("🔄 Load Documents", variant="primary")
            with gr.Row():  # before: (row did not exist)
                upload = gr.File(label="Upload PDF(s)", file_count="multiple", type="filepath", file_types=[".pdf"])  # before: (no upload control)
            with gr.Row():  # before: (row did not exist)
                upload_btn = gr.Button("📤 Upload PDFs", variant="secondary")  # before: (button did not exist)
                load_new_btn = gr.Button("🆕 Load New Documents", variant="secondary")  # before: (button did not exist)
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("**Quick Topics:**")
                    topic_suggestions = gr.Markdown("Load documents to see suggestions")
                
                with gr.Column(scale=1):
                    gr.Markdown("**System Status:**")
                    documents_info = gr.Markdown(
                        label="Loaded Documents",
                        height=400
                    )
            
            # Main interface tabs
            with gr.Tabs():
                
                # Q&A Tab
                with gr.TabItem("💬 Ask Questions"):
                    gr.Markdown("""
                    Ask questions about your textbook content. Klaro will provide answers based solely on 
                    the loaded documents with proper citations.
                    """)
                    
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        container=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            placeholder="Ask a question about your textbooks...",
                            show_label=False,
                            scale=4
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)
                    
                    gr.Examples(
                        examples=[
                            "What is photosynthesis?",
                            "Explain the structure of a cell",
                            "What are the main types of chemical bonds?",
                            "How does the digestive system work?"
                        ],
                        inputs=question_input
                    )
                
                # Summarization Tab
                with gr.TabItem("📝 Summarize Topics"):
                    gr.Markdown("""
                    Generate comprehensive summaries of topics across all your textbooks. 
                    Klaro will synthesize information from multiple sources.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            topic_input = gr.Textbox(
                                label="Topic to Summarize",
                                placeholder="Enter a topic (e.g., 'cell division', 'photosynthesis', 'chemical reactions')"
                            )
                            summarize_btn = gr.Button("Generate Summary", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("**Quick Topics:**")
                            topic_suggestions = gr.Markdown("Load documents to see suggestions")
                    
                    summary_output = gr.Markdown(
                        label="Summary",
                        height=400
                    )
                    
                    gr.Examples(
                        examples=[
                            "photosynthesis",
                            "cell structure",
                            "chemical bonds",
                            "digestive system",
                            "nervous system"
                        ],
                        inputs=topic_input
                    )
                
                # Search Tab
                with gr.TabItem("🔍 Search Documents"):
                    gr.Markdown("""
                    Search across all your documents to find relevant content. 
                    Use this to explore what information is available on specific topics.
                    """)
                    
                    with gr.Row():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="Enter keywords to search for..."
                        )
                        search_btn = gr.Button("Search", variant="primary")
                    
                    search_output = gr.Markdown(
                        label="Search Results",
                        height=500
                    )
                
                # System Info Tab
                with gr.TabItem("📊 System Information"):
                    gr.Markdown("""
                    View system statistics, loaded documents, and performance metrics.
                    """)
                    
                    with gr.Row():
                        refresh_stats_btn = gr.Button("🔄 Refresh Statistics", variant="secondary")
                    
                    with gr.Row():
                        with gr.Column():
                            stats_output = gr.Markdown(
                                label="System Statistics",
                                height=400
                            )
                        
                        with gr.Column():
                            documents_info = gr.Markdown(
                                label="Loaded Documents",
                                height=400
                            )
            
            # Event handlers
            
            # Load documents
            load_docs_btn.click(
                fn=self.load_documents_ui,
                inputs=[upload],  # before: no inputs
                outputs=[system_status, documents_info]
            ).then(
                fn=self.get_topic_suggestions_ui,
                outputs=topic_suggestions
            )

            upload_btn.click(  # before: (handler did not exist)
                fn=self.upload_files_ui,
                inputs=[upload],
                outputs=[system_status, documents_info]
            )

            load_new_btn.click(  # before: (handler did not exist)
                fn=self.load_new_documents_ui,
                inputs=[upload],
                outputs=[system_status, documents_info]
            ).then(
                fn=self.get_topic_suggestions_ui,
                outputs=topic_suggestions
            )
            
            # Q&A functionality
            question_input.submit(
                fn=self.ask_question_ui,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            ask_btn.click(
                fn=self.ask_question_ui,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            # Summarization functionality
            topic_input.submit(
                fn=self.summarize_topic_ui,
                inputs=topic_input,
                outputs=summary_output
            )
            
            summarize_btn.click(
                fn=self.summarize_topic_ui,
                inputs=topic_input,
                outputs=summary_output
            )
            
            # Search functionality
            search_input.submit(
                fn=self.search_documents_ui,
                inputs=search_input,
                outputs=search_output
            )
            
            search_btn.click(
                fn=self.search_documents_ui,
                inputs=search_input,
                outputs=search_output
            )
            
            # System statistics
            refresh_stats_btn.click(
                fn=self.get_system_stats_ui,
                outputs=stats_output
            )
            
            # Initialize with current stats
            interface.load(
                fn=self.get_system_stats_ui,
                outputs=stats_output
            )
        
        return interface
    
    def launch(self, 
               share: bool = False,
               server_name: str = "0.0.0.0",
               server_port: int = 7860,
               debug: bool = False) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            server_name: Server hostname
            server_port: Server port
            debug: Enable debug mode
        """
        interface = self.create_interface()
        
        logger.info(f"Launching Klaro UI on {server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            debug=debug,
            show_error=True,
            quiet=False
        )


def main():
    """Main entry point for the UI."""
    parser = argparse.ArgumentParser(description="Klaro Academic Chatbot UI")
    parser.add_argument("--docs-folder", default="./klaro_docs/", help="Path to documents folder")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    # mock disabled by default when llm path is provided
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM (override)")
    parser.add_argument("--llm-model", help="Path/name of local LLM model (disables mock)")  # before: (arg did not exist)
    
    args = parser.parse_args()
    
    try:
        # Create and launch UI
        ui = KlaroUI(
            docs_folder=args.docs_folder,
            llm_model_path=(args.llm_model or str(Path.home() / "mistral_models" / "7B-Instruct-v0.3")),  # default to downloaded path
            use_mock_llm=False if (args.llm_model or (Path.home() / "mistral_models" / "7B-Instruct-v0.3").exists()) else args.mock_llm
        )
        
        ui.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"Error launching UI: {str(e)}")
        raise


if __name__ == "__main__":
    main()

