"""
LLM Interface Module for Klaro Academic Chatbot

This module handles local language model integration, prompt engineering,
and response generation for the RAG pipeline. It provides a unified interface
for different local LLM models with educational content optimization.
"""

import logging
import time
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structure for LLM response with metadata"""
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    inference_time_ms: int
    model_name: str
    temperature: float
    confidence_score: Optional[float] = None

@dataclass
class PromptTemplate:
    """Structure for prompt templates"""
    name: str
    template: str
    description: str
    required_variables: List[str]

class LLMInterface:
    """
    Interface for local language model integration.
    
    Features:
    - Support for multiple local LLM models (Mistral, Phi-3, OpenChat)
    - Educational content-optimized prompt engineering
    - Response validation and filtering
    - Citation-aware response generation
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_name: str = "mistral-7b-instruct",
                 max_tokens: int = 512,
                 temperature: float = 0.1,
                 context_window: int = 4096):
        """
        Initialize the LLM interface.
        
        Args:
            model_path: Path to the GGUF model file
            model_name: Name identifier for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            context_window: Maximum context window size
        """
        self.model_path = model_path
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.context_window = context_window
        
        # Model instance (will be loaded lazily)
        self.model = None
        self.model_loaded = False
        
        # Performance tracking
        self.total_queries = 0
        self.total_inference_time = 0
        self.total_tokens_generated = 0
        
        # Initialize prompt templates
        self.prompt_templates = self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize educational prompt templates."""
        templates = {}
        
        # Q&A Template
        templates["qa"] = PromptTemplate(
            name="qa",
            template="""You are an educational assistant that answers questions based strictly on the provided textbook content. 

IMPORTANT RULES:
1. Only use information from the provided context
2. If the answer is not in the context, say "I couldn't find information about this in the provided materials"
3. Always cite your sources using the format [Source: filename, Page X]
4. Provide clear, educational explanations suitable for students
5. Do not add information from your general knowledge

Context:
{context}

Question: {question}

Answer:""",
            description="Template for answering questions based on textbook content",
            required_variables=["context", "question"]
        )
        
        # Summarization Template
        templates["summarize"] = PromptTemplate(
            name="summarize",
            template="""You are an educational assistant that creates comprehensive summaries of academic topics based on textbook content.

IMPORTANT RULES:
1. Only use information from the provided context
2. Create a well-structured summary with clear sections
3. Include citations for all major points using [Source: filename, Page X]
4. Make the summary educational and easy to understand
5. Organize information logically from basic concepts to more complex ones

Context from multiple sources:
{context}

Topic to summarize: {topic}

Please provide a comprehensive summary of this topic based on the provided materials:""",
            description="Template for creating topic summaries from multiple sources",
            required_variables=["context", "topic"]
        )
        
        # Explanation Template
        templates["explain"] = PromptTemplate(
            name="explain",
            template="""You are an educational tutor that provides detailed explanations of concepts based on textbook content.

IMPORTANT RULES:
1. Only use information from the provided context
2. Break down complex concepts into understandable parts
3. Use examples from the textbook when available
4. Cite sources for all information using [Source: filename, Page X]
5. Structure your explanation clearly with logical flow

Context:
{context}

Concept to explain: {concept}

Please provide a detailed educational explanation:""",
            description="Template for explaining concepts in detail",
            required_variables=["context", "concept"]
        )
        
        return templates
    
    def load_model(self) -> bool:
        """
        Load the language model (Mistral Inference, local).
        """
        if self.model_loaded:
            return True

        try:
            from mistral_inference import Mistral  # before: from llama_cpp import Llama
            model_dir = Path(self.model_path or "")
            if not model_dir.exists():
                logger.error(f"Model directory not found: {model_dir}")
                return False

            # Required files downloaded by snapshot_download
            params = model_dir / "params.json"
            weights = model_dir / "consolidated.safetensors"
            tokenizer = model_dir / "tokenizer.model.v3"
            for p in (params, weights, tokenizer):
                if not p.exists():
                    logger.error(f"Missing model file: {p}")
                    return False

            # Initialize Mistral (local directory)
            # Note: mistral_inference accepts the directory path.
            self.model = Mistral(model_dir=str(model_dir))  # before: Llama(model_path=...)
            self.model_loaded = True
            logger.info(f"Successfully loaded Mistral model: {self.model_name} at {model_dir}")
            return True

        except ImportError:
            logger.error("mistral_inference not installed. Run: pip install mistral_inference")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         stop_sequences: Optional[List[str]] = None) -> LLMResponse:
        """
        Generate a response using the loaded model.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            stop_sequences: List of sequences that stop generation
            
        Returns:
            LLMResponse with generated text and metadata
        """
        if not self.model_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load language model")
        
        # Use provided parameters or defaults
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        stop_sequences = stop_sequences or ["Human:", "User:", "\n\n---"]
        
        start_time = time.time()
        
        try:
            # Generate response via mistral_inference
            # Some builds expose .generate(prompt, max_tokens=..., temperature=..., stop=...)
            output_text = self.model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
            )
            # If API returns an object, try to extract text:
            if isinstance(output_text, dict):
                output_text = output_text.get("text") or output_text.get("choices", [{}])[0].get("text", "")

            response_text = (output_text or "").strip()
            inference_time = int((time.time() - start_time) * 1000)

            # Token counts (approx)
            prompt_tokens = int(len(prompt.split()) * 1.3)
            completion_tokens = int(len(response_text.split()) * 1.3)
            total_tokens = prompt_tokens + completion_tokens

            self.total_queries += 1
            self.total_inference_time += inference_time
            self.total_tokens_generated += completion_tokens

            return LLMResponse(
                response_text=response_text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                inference_time_ms=inference_time,
                model_name=self.model_name,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise RuntimeError(f"Failed to generate response: {e}")
    
    def answer_question(self, 
                       question: str, 
                       context: str) -> LLMResponse:
        """
        Answer a question based on provided context.
        
        Args:
            question: User question
            context: Retrieved context from documents
            
        Returns:
            LLMResponse with the answer
        """
        template = self.prompt_templates["qa"]
        prompt = template.template.format(
            context=context,
            question=question
        )
        
        # Ensure prompt fits in context window
        prompt = self._truncate_prompt(prompt)
        
        return self.generate_response(
            prompt,
            temperature=0.1,  # Low temperature for factual answers
            stop_sequences=["Question:", "Context:", "\n\n---"]
        )
    
    def summarize_topic(self, 
                       topic: str, 
                       context: str) -> LLMResponse:
        """
        Generate a summary of a topic based on provided context.
        
        Args:
            topic: Topic to summarize
            context: Retrieved context from multiple documents
            
        Returns:
            LLMResponse with the summary
        """
        template = self.prompt_templates["summarize"]
        prompt = template.template.format(
            context=context,
            topic=topic
        )
        
        # Ensure prompt fits in context window
        prompt = self._truncate_prompt(prompt)
        
        return self.generate_response(
            prompt,
            max_tokens=1024,  # Longer responses for summaries
            temperature=0.2,  # Slightly higher for more natural language
            stop_sequences=["Topic:", "Context:", "\n\n---"]
        )
    
    def explain_concept(self, 
                       concept: str, 
                       context: str) -> LLMResponse:
        """
        Provide a detailed explanation of a concept.
        
        Args:
            concept: Concept to explain
            context: Retrieved context from documents
            
        Returns:
            LLMResponse with the explanation
        """
        template = self.prompt_templates["explain"]
        prompt = template.template.format(
            context=context,
            concept=concept
        )
        
        # Ensure prompt fits in context window
        prompt = self._truncate_prompt(prompt)
        
        return self.generate_response(
            prompt,
            max_tokens=768,
            temperature=0.15,
            stop_sequences=["Concept:", "Context:", "\n\n---"]
        )
    
    def _truncate_prompt(self, prompt: str) -> str:
        """
        Truncate prompt to fit within context window.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Truncated prompt that fits in context window
        """
        # Rough token estimation (1 token ≈ 0.75 words)
        estimated_tokens = len(prompt.split()) * 1.3
        
        if estimated_tokens <= self.context_window * 0.8:  # Leave room for response
            return prompt
        
        # Find the context section and truncate it
        context_start = prompt.find("Context:")
        context_end = prompt.find("\n\nQuestion:") or prompt.find("\n\nTopic:") or prompt.find("\n\nConcept:")
        
        if context_start != -1 and context_end != -1:
            # Calculate how much context we can keep
            non_context_length = len(prompt) - (context_end - context_start)
            max_context_length = int((self.context_window * 0.8 * 0.75) - non_context_length)  # Convert tokens to chars
            
            if max_context_length > 500:  # Ensure we have meaningful context
                context_section = prompt[context_start:context_end]
                truncated_context = context_section[:max_context_length] + "\n\n[Context truncated for length]"
                
                prompt = prompt[:context_start] + truncated_context + prompt[context_end:]
        
        return prompt
    
    def validate_response(self, response: str, context: str) -> Tuple[bool, List[str]]:
        """
        Validate that response is grounded in the provided context.
        
        Args:
            response: Generated response text
            context: Source context
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "i know that",
            "it is well known",
            "generally speaking",
            "in my experience",
            "based on my knowledge"
        ]
        
        response_lower = response.lower()
        for phrase in hallucination_phrases:
            if phrase in response_lower:
                issues.append(f"Potential hallucination: '{phrase}' suggests external knowledge")
        
        # Check for proper citation format
        citation_pattern = r'\[Source: [^,]+, Page \d+\]'
        citations = re.findall(citation_pattern, response)
        
        if not citations and len(response) > 100:
            issues.append("Response lacks proper citations")
        
        # Check if response acknowledges lack of information appropriately
        if "couldn't find" in response_lower or "not in the provided" in response_lower:
            # This is good - the model is being honest about limitations
            pass
        elif len(context.strip()) < 100 and len(response) > 200:
            issues.append("Response seems too detailed for limited context")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the LLM interface.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_inference_time = (
            self.total_inference_time / self.total_queries 
            if self.total_queries > 0 else 0
        )
        
        avg_tokens_per_query = (
            self.total_tokens_generated / self.total_queries 
            if self.total_queries > 0 else 0
        )
        
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "total_queries": self.total_queries,
            "total_inference_time_ms": self.total_inference_time,
            "average_inference_time_ms": round(avg_inference_time, 2),
            "total_tokens_generated": int(self.total_tokens_generated),
            "average_tokens_per_query": round(avg_tokens_per_query, 1),
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def list_available_templates(self) -> List[Dict[str, str]]:
        """
        List available prompt templates.
        
        Returns:
            List of template information
        """
        return [
            {
                "name": template.name,
                "description": template.description,
                "required_variables": template.required_variables
            }
            for template in self.prompt_templates.values()
        ]


# Mock LLM Interface for testing without actual model
class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing and development without actual models."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_loaded = True  # Always "loaded"
    
    def load_model(self) -> bool:
        """Mock model loading."""
        self.model_loaded = True
        return True
    
    def generate_response(self, 
                         prompt: str, 
                         max_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         stop_sequences: Optional[List[str]] = None) -> LLMResponse:
        """Generate mock response."""
        time.sleep(0.1)  # Simulate inference time
        
        # Generate a simple mock response based on prompt content
        if "question:" in prompt.lower():
            response_text = "Based on the provided context, this is a mock educational response that would normally be generated by the language model. [Source: example.pdf, Page 1]"
        elif "summarize" in prompt.lower():
            response_text = "This is a mock summary of the requested topic based on the provided educational materials. The summary would include key concepts and proper citations. [Source: textbook.pdf, Page 5]"
        else:
            response_text = "This is a mock response from the educational assistant. In a real implementation, this would be generated by the local language model."
        
        return LLMResponse(
            response_text=response_text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(prompt.split()) + len(response_text.split()),
            inference_time_ms=100,
            model_name="mock-model",
            temperature=temperature or self.temperature
        )


# Example usage and testing
if __name__ == "__main__":
    # Test with mock interface
    llm = MockLLMInterface(model_name="mock-mistral-7b")
    
    # Test question answering
    context = """
    Biology is the scientific study of life and living organisms. 
    It encompasses many fields including molecular biology, genetics, and ecology.
    Cells are the basic units of life and contain organelles like the nucleus and mitochondria.
    """
    
    question = "What is biology?"
    
    response = llm.answer_question(question, context)
    
    print(f"Question: {question}")
    print(f"Response: {response.response_text}")
    print(f"Inference time: {response.inference_time_ms}ms")
    print(f"Tokens: {response.total_tokens}")
    
    # Validate response
    is_valid, issues = llm.validate_response(response.response_text, context)
    print(f"Response valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    
    # Test summarization
    topic = "cell structure"
    summary_response = llm.summarize_topic(topic, context)
    print(f"\nSummary of '{topic}':")
    print(summary_response.response_text)
    
    # Print performance stats
    stats = llm.get_performance_stats()
    print(f"\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    
    # List templates
    templates = llm.list_available_templates()
    print(f"\nAvailable Templates:")
    for template in templates:
        print(f"- {template['name']}: {template['description']}")

