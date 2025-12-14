"""
rag_utils_simple.py
Minimal RAG utilities for local / Docker CPU usage
"""

import torch
import logging
from typing import List, Tuple, Dict, Any

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIG
# =============================================================================

MODEL_NAME = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Context limits
MAX_CONTEXT_CHARS = 2000  # Increased for more context

# =============================================================================
# GLOBAL MODEL (LOAD ONCE)
# =============================================================================

tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if model is not None:
        return

    logger.info(f"Loading model: {MODEL_NAME} on {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    model.eval()
    logger.info("Model loaded successfully")


# =============================================================================
# 1. CONTEXT PREPARATION (OPENSEARCH-AWARE)
# =============================================================================

def prepare_context(
    results: List[Tuple[Any, float]],
    max_chars: int = MAX_CONTEXT_CHARS
) -> Dict[str, Any]:
    """
    Prepare context for FLAN-T5.
    
    OpenSearch with FAISS/L2: LOWER score = BETTER match
    """
    
    logger.info(f"Preparing context from {len(results)} results")

    if not results:
        logger.warning("No results provided")
        return {"context": "", "sources": [], "is_empty": True}

    # Sort by similarity (lower score = better for L2 distance)
    sorted_results = sorted(results, key=lambda x: x[1])
    
    # Log scores for debugging
    for i, (doc, score) in enumerate(sorted_results[:3]):
        logger.info(f"Result {i+1}: score={score:.4f}, content_length={len(doc.page_content)}")
        logger.info(f"Content preview: {doc.page_content[:150]}...")

    context_parts = []
    sources = []
    total_chars = 0

    for i, (doc, score) in enumerate(sorted_results):
        text = doc.page_content.strip()
        
        if not text or len(text) < 50:  # Skip very short snippets
            logger.info(f"Skipping short snippet: {len(text)} chars")
            continue

        remaining = max_chars - total_chars
        if remaining <= 200:  # Need at least 200 chars for meaningful content
            break

        # Take up to remaining characters
        snippet = text[:remaining]
        
        # Don't add source labels - just the content
        context_parts.append(snippet)
        total_chars += len(snippet)

        sources.append({
            "id": i + 1,
            "url": doc.metadata.get("source", "unknown"),
            "score": float(score),
            "preview": text[:200] + "..." if len(text) > 200 else text
        })

    final_context = "\n\n".join(context_parts)
    
    logger.info(f"Context prepared: {len(context_parts)} parts, {total_chars} chars")
    logger.info(f"Final context preview: {final_context[:300]}...")

    return {
        "context": final_context,
        "sources": sources,
        "is_empty": len(context_parts) == 0
    }


# =============================================================================
# 2. PROMPT (FLAN-T5 OPTIMIZED)
# =============================================================================

def construct_prompt(query: str, context: str) -> str:
    """
    Improved prompt for FLAN-T5
    """
    # More explicit instructions
    prompt = f"""Read the following information and answer the question.

Information:
{context}

Question: {query}

Provide a clear and detailed answer based only on the information above:"""
    
    logger.info(f"Prompt length: {len(prompt)} chars")
    logger.info(f"Full prompt:\n{prompt}")
    
    return prompt.strip()


# =============================================================================
# 3. GENERATION
# =============================================================================

def generate_answer(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.3,  # Lower temperature for more focused answers
    top_p: float = 0.85
) -> str:
    """
    Generate answer using FLAN-T5
    """
    load_model()

    logger.info("Generating answer with FLAN-T5...")

    # Tokenize with proper attention to length
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False
    ).to(DEVICE)

    logger.info(f"Input tokens: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,  # Ensure at least some output
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=3,  # Reduced for speed
            early_stopping=True,
            repetition_penalty=1.2,  # Reduce repetition
            no_repeat_ngram_size=3
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    logger.info(f"Generated answer ({len(answer)} chars): {answer}")
    
    return answer


# =============================================================================
# 4. FINAL RESPONSE
# =============================================================================

def process_response(
    answer: str,
    query: str,
    sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Format final response with fallback handling
    """
    
    # Check if answer is weak or nonsensical
    weak_indicators = [
        len(answer) < 15,
        answer.strip() in ["(a)", "(b)", "(c)", "yes", "no"],
        "don't know" in answer.lower(),
        "cannot" in answer.lower() and len(answer) < 30
    ]
    
    if any(weak_indicators):
        logger.warning(f"Weak answer detected: '{answer}'")
        
        # Try to extract info from sources as fallback
        if sources and len(sources) > 0:
            # Return the preview of the best matching source
            best_source = sources[0]
            answer = f"Based on the available information: {best_source['preview']}"
            logger.info(f"Using fallback answer from best source")
    
    return {
        "answer": answer,
        "query": query,
        "sources": sources,
        "num_sources": len(sources)
    }