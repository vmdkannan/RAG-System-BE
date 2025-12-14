"""
rag_utils.py - RAG Pipeline Utility Functions

Modular functions for:
1. Context Preparation
2. Prompt Construction
3. LLM Generation (HuggingFace)
4. Response Processing

Usage:
    from rag_utils import prepare_context, construct_prompt, generate_with_huggingface, process_response
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# 1. CONTEXT PREPARATION
# ============================================================================

def prepare_context(
    results: List[Tuple[Any, float]], 
    score_threshold: float = 0.3,
    max_context_length: int = 4000,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Prepare and filter context from vector search results.
    
    Args:
        results: List of tuples (document, score) from vector search
        score_threshold: Minimum similarity score to include
        max_context_length: Maximum character length for context
        include_metadata: Whether to include source metadata
    
    Returns:
        Dictionary containing formatted context and source information
        {
            'context': str,
            'sources': List[Dict],
            'num_sources': int,
            'is_empty': bool,
            'total_characters': int,
            'filtered_count': int,
            'included_count': int
        }
    """
    try:
        # Filter results by score threshold
        filtered_results = []
        for doc, score in results:
            if score >= score_threshold:
                filtered_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
        
        if not filtered_results:
            return {
                "context": "",
                "sources": [],
                "num_sources": 0,
                "is_empty": True,
                "message": "No relevant documents found above the score threshold.",
                "total_characters": 0,
                "filtered_count": 0,
                "included_count": 0
            }
        
        # Sort by score (highest first)
        filtered_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Build context string with character limit
        context_parts = []
        sources = []
        current_length = 0
        
        for i, result in enumerate(filtered_results):
            content = result['content']
            score = result['score']
            metadata = result['metadata']
            
            # Format source entry
            source_header = f"\n--- Source {i+1} (Relevance: {score:.2f}) ---\n"
            if include_metadata and metadata:
                source_info = f"Source: {metadata.get('source', 'Unknown')}\n"
                if metadata.get('type'):
                    source_info += f"Type: {metadata.get('type')}\n"
                source_header += source_info
            
            # Check if adding this would exceed limit
            entry = f"{source_header}{content}\n"
            if current_length + len(entry) > max_context_length:
                logger.info(f"Reached context length limit. Including {i} out of {len(filtered_results)} sources.")
                break
            
            context_parts.append(entry)
            sources.append({
                "source_id": i + 1,
                "url": metadata.get('source', 'Unknown'),
                "type": metadata.get('type', 'Unknown'),
                "score": score,
                "preview": content[:200] + "..." if len(content) > 200 else content
            })
            current_length += len(entry)
        
        # Join all context parts
        full_context = "\n".join(context_parts)
        
        return {
            "context": full_context,
            "sources": sources,
            "num_sources": len(sources),
            "is_empty": False,
            "total_characters": current_length,
            "filtered_count": len(filtered_results),
            "included_count": len(sources)
        }
    
    except Exception as e:
        logger.error(f"Error preparing context: {str(e)}", exc_info=True)
        return {
            "context": "",
            "sources": [],
            "num_sources": 0,
            "is_empty": True,
            "error": str(e),
            "total_characters": 0,
            "filtered_count": 0,
            "included_count": 0
        }


# ============================================================================
# 2. PROMPT CONSTRUCTION
# ============================================================================

def construct_prompt(
    query: str,
    context: str,
    prompt_template: str = "default",
    custom_instructions: Optional[str] = None
) -> str:
    """
    Construct a prompt for the LLM using the query and context.
    
    Args:
        query: User's question
        context: Retrieved context from vector store
        prompt_template: Template type ('default', 'concise', 'detailed', 'custom')
        custom_instructions: Custom instructions to add to the prompt
    
    Returns:
        Formatted prompt string
    """
    try:
        # Define prompt templates
        templates = {
            "default": """You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Instructions:
- Use only the information from the context above
- If the context doesn't contain enough information, say so clearly
- Provide accurate and concise answers
- Cite specific sources when making claims

Question: {query}

Answer:""",
            
            "concise": """Based on the following context, provide a concise answer to the question.

Context:
{context}

Question: {query}

Concise Answer:""",
            
            "detailed": """You are an expert analyst. Provide a comprehensive answer based on the context below.

Context Information:
{context}

Analysis Requirements:
- Provide a detailed, well-structured answer
- Support claims with specific information from the sources
- Note any limitations or gaps in the available information
- Organize your response with clear sections if needed

User Question: {query}

Detailed Analysis:""",
            
            "qa": """Context: {context}

Question: {query}

Answer (based only on the context above):""",

            "custom": """{custom_instructions}

Context:
{context}

Question: {query}

Answer:"""
        }
        
        # Select template
        if prompt_template not in templates:
            logger.warning(f"Unknown template '{prompt_template}', using 'default'")
            prompt_template = "default"
        
        template = templates[prompt_template]
        
        # Format the prompt
        if prompt_template == "custom" and custom_instructions:
            prompt = template.format(
                custom_instructions=custom_instructions,
                context=context,
                query=query
            )
        else:
            prompt = template.format(
                context=context,
                query=query
            )
        
        return prompt
    
    except Exception as e:
        logger.error(f"Error constructing prompt: {str(e)}", exc_info=True)
        # Return a basic fallback prompt
        return f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"


# ============================================================================
# 3. LLM GENERATION (HUGGINGFACE)
# ============================================================================

class HuggingFaceLLM:
    """
    Wrapper class for HuggingFace LLM generation.
    Singleton pattern to avoid reloading models.
    """
    
    _instances = {}
    
    def __new__(cls, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", *args, **kwargs):
        """Implement singleton pattern per model"""
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[model_name] = instance
        return cls._instances[model_name]
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            load_in_8bit: Whether to load model in 8-bit precision
            use_auth_token: HuggingFace authentication token
        """
        # Only initialize once
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.load_in_8bit = load_in_8bit
        self.use_auth_token = use_auth_token
        self.initialized = False
        
        logger.info(f"Initializing HuggingFace LLM: {model_name} on {self.device}")
    
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            if self.initialized and self.model is not None:
                logger.info("Model already loaded")
                return
            
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.use_auth_token
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loading model {self.model_name}...")
            if self.load_in_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    use_auth_token=self.use_auth_token
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_auth_token=self.use_auth_token
                )
                if not self.load_in_8bit:
                    self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self.initialized = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        Generate response using the loaded model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repetition
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text (without the input prompt)
        """
        try:
            if not self.initialized or self.pipeline is None:
                self.load_model()
            
            logger.info("Generating response...")
            
            # Generate
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,  # Only return generated text
                **kwargs
            )
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
            else:
                generated_text = ""
            
            response = generated_text.strip()
            
            logger.info(f"Generated response length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise


def generate_with_huggingface(
    prompt: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Standalone function to generate response using HuggingFace.
    
    Args:
        prompt: Input prompt
        model_name: HuggingFace model identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to use ('cuda', 'cpu', or None for auto)
        **kwargs: Additional generation parameters
    
    Returns:
        Dictionary with generated response and metadata
        {
            'response': str,
            'model': str,
            'success': bool,
            'error': Optional[str],
            'generation_time': float
        }
    """
    try:
        start_time = datetime.now()
        
        llm = HuggingFaceLLM(model_name=model_name, device=device)
        llm.load_model()
        
        response = llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        return {
            "response": response,
            "model": model_name,
            "success": True,
            "error": None,
            "generation_time": generation_time
        }
        
    except Exception as e:
        logger.error(f"Error in HuggingFace generation: {str(e)}", exc_info=True)
        return {
            "response": "",
            "model": model_name,
            "success": False,
            "error": str(e),
            "generation_time": 0.0
        }


# ============================================================================
# 4. RESPONSE PROCESSING
# ============================================================================

def process_response(
    raw_response: str,
    sources: List[Dict[str, Any]],
    query: str,
    include_citations: bool = True,
    format_markdown: bool = True,
    add_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Process and format the LLM response.
    
    Args:
        raw_response: Raw response from LLM
        sources: List of source documents
        query: Original user query
        include_citations: Whether to add source citations
        format_markdown: Whether to format as markdown
        add_timestamp: Whether to add timestamp
    
    Returns:
        Processed response dictionary
        {
            'answer': str,
            'raw_answer': str,
            'query': str,
            'sources': List[Dict],
            'num_sources': int,
            'metrics': Dict,
            'quality_checks': Dict,
            'is_quality_response': bool,
            'timestamp': Optional[str]
        }
    """
    try:
        # Clean the response
        cleaned_response = raw_response.strip()
        
        # Remove common artifacts
        artifacts_to_remove = [
            "Answer:",
            "Response:",
            "Based on the context,",
            "According to the provided information,",
            "Concise Answer:",
            "Detailed Analysis:"
        ]
        for artifact in artifacts_to_remove:
            if cleaned_response.startswith(artifact):
                cleaned_response = cleaned_response[len(artifact):].strip()
        
        # Format response
        if format_markdown:
            formatted_response = cleaned_response
            
            # Add source citations if requested
            if include_citations and sources:
                citations = "\n\n---\n\n### Sources\n\n"
                for i, source in enumerate(sources):
                    source_url = source.get('url', 'Unknown')
                    source_score = source.get('score', 0)
                    citations += f"{i+1}. [{source_url}]({source_url}) "
                    citations += f"(Relevance: {source_score:.2f})\n"
                
                formatted_response += citations
        else:
            formatted_response = cleaned_response
        
        # Calculate response metrics
        word_count = len(cleaned_response.split())
        char_count = len(cleaned_response)
        sentence_count = cleaned_response.count('.') + cleaned_response.count('!') + cleaned_response.count('?')
        
        # Check response quality
        quality_checks = {
            "has_content": len(cleaned_response) > 10,
            "not_too_short": word_count >= 5,
            "not_placeholder": not any(phrase in cleaned_response.lower() for phrase in [
                "i don't know",
                "i cannot answer",
                "no information available"
            ]) or word_count > 15,
            "has_structure": sentence_count > 0
        }
        
        is_quality_response = all(quality_checks.values())
        
        # Add timestamp if requested
        timestamp = datetime.now().isoformat() if add_timestamp else None
        
        return {
            "answer": formatted_response,
            "raw_answer": raw_response,
            "query": query,
            "sources": sources,
            "num_sources": len(sources),
            "metrics": {
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentence_count,
                "quality_score": sum(quality_checks.values()) / len(quality_checks)
            },
            "quality_checks": quality_checks,
            "is_quality_response": is_quality_response,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.error(f"Error processing response: {str(e)}", exc_info=True)
        return {
            "answer": raw_response,
            "raw_answer": raw_response,
            "query": query,
            "sources": sources,
            "num_sources": len(sources),
            "error": str(e),
            "is_quality_response": False,
            "timestamp": datetime.now().isoformat() if add_timestamp else None
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_models() -> List[str]:
    """
    Get list of recommended HuggingFace models for RAG.
    
    Returns:
        List of model names
    """
    return [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "google/flan-t5-large",
        "tiiuae/falcon-7b-instruct",
        "stabilityai/stablelm-tuned-alpha-7b"
    ]


def validate_context(context_data: Dict[str, Any]) -> bool:
    """
    Validate context data structure.
    
    Args:
        context_data: Output from prepare_context()
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['context', 'sources', 'is_empty']
    return all(key in context_data for key in required_keys)