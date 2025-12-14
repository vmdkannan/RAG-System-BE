from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
import re
from collections import Counter
from bs4 import BeautifulSoup

from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from rag_utils_simple import (
    prepare_context,
    construct_prompt,
    generate_answer,
    process_response
)

# =============================================================================
# CONFIG
# =============================================================================

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX = "default_index"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag-app")

# =============================================================================
# APP
# =============================================================================

app = Flask(__name__)
CORS(app)

# =============================================================================
# EMBEDDINGS
# =============================================================================

logger.info("Loading embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise

# =============================================================================
# OPENSEARCH CONNECTION
# =============================================================================

try:
    if ENVIRONMENT == "local":
        opensearch_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=None,
            use_ssl=False,
            verify_certs=False,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        opensearch_url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"

    elif ENVIRONMENT == "aws":
        opensearch_client = OpenSearch(
            hosts=[{"host": os.getenv("OPENSEARCH_ENDPOINT"), "port": 443}],
            http_auth=(
                os.getenv("OPENSEARCH_USER"),
                os.getenv("OPENSEARCH_PASSWORD")
            ),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        opensearch_url = f"https://{os.getenv('OPENSEARCH_ENDPOINT')}:443"

    logger.info(f"OpenSearch configured for {ENVIRONMENT} at {opensearch_url}")
    health = opensearch_client.cluster.health()
    logger.info(f"OpenSearch cluster status: {health.get('status')}")

except Exception as e:
    logger.error(f"Failed to connect to OpenSearch: {str(e)}")
    raise

# =============================================================================
# HELPER: EXTRACT KEYWORDS
# =============================================================================

def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Simple keyword extraction based on frequency
    """
    # Common stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once'
    }
    
    # Extract words (keep only alphabetic, 3+ chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stopwords
    words = [w for w in words if w not in stopwords]
    
    # Count frequencies
    counter = Counter(words)
    
    # Return top N
    return [word for word, _ in counter.most_common(top_n)]

# =============================================================================
# HELPER: CLEAN WEB CONTENT
# =============================================================================

def clean_web_content(url: str):
    """Extract readable text from a web page"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe", "form"]):
            tag.decompose()

        # Try to find main content
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=["content", "article", "post", "story"])
            or soup.find("body")
        )

        if not main:
            return None

        text = main.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return None

# =============================================================================
# HEALTH
# =============================================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        health = opensearch_client.cluster.health()
        return jsonify({
            "status": "healthy",
            "opensearch": {
                "status": health.get("status"),
                "cluster_name": health.get("cluster_name"),
                "nodes": health.get("number_of_nodes")
            },
            "embedding_model": EMBEDDING_MODEL
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

# =============================================================================
# PROCESS URLS - IMPROVED WITH BETTER CHUNKING & METADATA
# =============================================================================

@app.route("/process-urls", methods=["POST"])
def process_urls():
    """
    Process URLs with improved chunking, filtering, and metadata
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        urls = data.get("urls", [])
        index_name = data.get("store_id", DEFAULT_INDEX).lower().replace(" ", "_")
        chunk_size = data.get("chunk_size", 800)  # Smaller for more focused chunks
        chunk_overlap = data.get("chunk_overlap", 150)

        if not urls or not isinstance(urls, list):
            return jsonify({"error": "Valid URLs list required"}), 400

        logger.info(f"Processing {len(urls)} URLs for index: {index_name}")

        # Extract content from URLs
        documents = []
        for url in urls:
            logger.info(f"Processing: {url}")
            text = clean_web_content(url)
            
            if text and len(text) > 100:
                # Extract title and keywords for better metadata
                lines = text.split('\n')
                title = lines[0] if lines else "Unknown"
                
                # Extract keywords from content
                keywords = extract_keywords(text, top_n=10)
                
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "type": "web",
                            "title": title[:200],  # First 200 chars as title
                            "keywords": ", ".join(keywords)  # Store keywords
                        }
                    )
                )
                logger.info(f"✓ Extracted {len(text)} characters from {url}")
                logger.info(f"  Title: {title[:100]}")
                logger.info(f"  Keywords: {', '.join(keywords[:5])}")
            else:
                logger.warning(f"✗ Insufficient content from {url}")

        if not documents:
            return jsonify({"error": "No usable content extracted from URLs"}), 400

        logger.info(f"Successfully extracted content from {len(documents)} URLs")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len
        )

        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} initial chunks")

        # IMPROVED FILTERING - Remove low-quality chunks
        quality_chunks = []
        for chunk in chunks:
            text = chunk.page_content.strip()
            
            # Skip if too short
            if len(text) < 100:
                continue
            
            # Skip if too many line breaks (likely navigation)
            if text.count('\n') / max(len(text), 1) > 0.1:
                continue
            
            # Skip if starts with common navigation patterns
            skip_patterns = [
                'http', 'www', 'api_url', 'Home\nNews\nBusiness',
                'Trending Topics', 'Subscribe', 'Follow us', 'Share',
                'Click here', 'Read more', 'Sign in', 'Sign up',
                'Menu', 'Search', 'Contact', 'About us'
            ]
            if any(text.lower().startswith(pattern.lower()) for pattern in skip_patterns):
                continue
            
            # Skip if mostly short words (likely navigation)
            words = text.split()
            if len(words) > 0:
                avg_word_length = sum(len(w) for w in words) / len(words)
                if avg_word_length < 4:  # Suspiciously short
                    continue
            
            # Skip if too many repeated characters (likely formatting issues)
            if re.search(r'(.)\1{5,}', text):
                continue
            
            quality_chunks.append(chunk)

        logger.info(f"Filtered to {len(quality_chunks)} quality chunks")

        if not quality_chunks:
            return jsonify({"error": "No quality chunks generated"}), 400

        # Create vector store
        logger.info("Creating vector store in OpenSearch...")
        
        OpenSearchVectorSearch.from_documents(
            documents=quality_chunks,
            embedding=embedding_model,
            opensearch_url=opensearch_url,
            index_name=index_name,
            http_auth=None if ENVIRONMENT == "local" else (
                os.getenv("OPENSEARCH_USER"),
                os.getenv("OPENSEARCH_PASSWORD")
            ),
            use_ssl=ENVIRONMENT != "local",
            verify_certs=ENVIRONMENT != "local",
            connection_class=RequestsHttpConnection,
            engine="faiss",
            space_type="l2",
            bulk_size=500
        )

        logger.info(f"Vector store '{index_name}' created successfully")

        return jsonify({
            "status": "success",
            "index_name": index_name,
            "urls_processed": len(urls),
            "documents": len(documents),
            "chunks": len(quality_chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }), 200

    except Exception as e:
        logger.error("Error processing URLs", exc_info=True)
        return jsonify({
            "error": "Failed to process URLs",
            "details": str(e)
        }), 500

# =============================================================================
# RAG ASK - IMPROVED WITH RE-RANKING
# =============================================================================

@app.route("/rag/ask", methods=["POST"])
def ask():
    """
    Query with improved relevance scoring and re-ranking
    """
    try:
        data = request.json or {}
        
        query = data.get("query", "").strip()
        index_name = data.get("store_id", DEFAULT_INDEX).lower().replace(" ", "_")
        k = int(data.get("k", 10))  # Get more results initially for re-ranking

        if not query:
            return jsonify({"error": "query is required"}), 400

        logger.info(f"RAG query: '{query}' on index: '{index_name}' (k={k})")

        # Check if index exists
        if not opensearch_client.indices.exists(index=index_name):
            available = [
                idx for idx in opensearch_client.indices.get_alias(index="*").keys() 
                if not idx.startswith('.')
            ]
            return jsonify({
                "error": f"Index '{index_name}' not found",
                "available_indices": available
            }), 404

        # Perform vector search
        vectorstore = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding_model,
            http_auth=None if ENVIRONMENT == "local" else (
                os.getenv("OPENSEARCH_USER"),
                os.getenv("OPENSEARCH_PASSWORD")
            ),
            use_ssl=ENVIRONMENT != "local",
            verify_certs=ENVIRONMENT != "local",
            connection_class=RequestsHttpConnection
        )

        logger.info("Performing similarity search...")
        results = vectorstore.similarity_search_with_score(query=query, k=k)
        logger.info(f"Found {len(results)} results")

        # IMPROVED: Re-rank results based on keyword matching
        query_keywords = set(extract_keywords(query, top_n=5))
        logger.info(f"Query keywords: {query_keywords}")
        
        ranked_results = []
        for doc, vector_score in results:
            # Calculate keyword overlap
            doc_text = doc.page_content.lower()
            doc_keywords = set(extract_keywords(doc.page_content, top_n=10))
            
            # Count matching keywords
            keyword_matches = len(query_keywords.intersection(doc_keywords))
            
            # Also check for exact phrase matches
            query_words = query.lower().split()
            phrase_matches = sum(1 for word in query_words if word in doc_text)
            
            # Combined score (lower vector score is better)
            # Boost documents with more keyword matches
            keyword_boost = keyword_matches * 0.05  # Reduce vector score by 0.05 per match
            phrase_boost = phrase_matches * 0.02    # Additional boost for phrase matches
            
            combined_score = vector_score - keyword_boost - phrase_boost
            
            ranked_results.append({
                'doc': doc,
                'combined_score': combined_score,
                'vector_score': vector_score,
                'keyword_matches': keyword_matches,
                'phrase_matches': phrase_matches
            })
            
        # Sort by combined score (lower is better)
        ranked_results.sort(key=lambda x: x['combined_score'])
        
        # Log top results for debugging
        logger.info("Top ranked results:")
        for i, result in enumerate(ranked_results[:3]):
            logger.info(f"  Rank {i+1}:")
            logger.info(f"    Vector: {result['vector_score']:.4f}")
            logger.info(f"    Keywords: {result['keyword_matches']}")
            logger.info(f"    Phrases: {result['phrase_matches']}")
            logger.info(f"    Combined: {result['combined_score']:.4f}")
            logger.info(f"    Preview: {result['doc'].page_content[:150]}...")
        
        # Take top 5 results after re-ranking
        final_results = [(r['doc'], r['combined_score']) for r in ranked_results[:5]]

        # Prepare context
        context_data = prepare_context(final_results)

        if context_data["is_empty"]:
            logger.warning("Context is empty!")
            return jsonify({
                "answer": "No relevant information found in the knowledge base.",
                "query": query,
                "sources": [],
                "debug": {
                    "num_results": len(results),
                    "index": index_name,
                    "query_keywords": list(query_keywords)
                }
            }), 200

        # Generate answer
        prompt = construct_prompt(
            query=query,
            context=context_data["context"]
        )

        logger.info("Generating answer with LLM...")
        answer = generate_answer(prompt)

        # Process response
        response = process_response(
            answer=answer,
            query=query,
            sources=context_data["sources"]
        )

        logger.info("RAG response generated successfully")
        return jsonify(response), 200

    except Exception as e:
        logger.error("RAG ask failed", exc_info=True)
        return jsonify({
            "error": "RAG query failed",
            "details": str(e)
        }), 500

# =============================================================================
# DEBUG ENDPOINTS
# =============================================================================

@app.route("/debug/search", methods=["POST"])
def debug_search():
    """Debug endpoint to see raw vector search results with re-ranking"""
    try:
        data = request.json or {}
        query = data.get("query", "").strip()
        index_name = data.get("store_id", DEFAULT_INDEX).lower().replace(" ", "_")
        k = int(data.get("k", 10))

        if not query:
            return jsonify({"error": "query required"}), 400

        vectorstore = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding_model,
            http_auth=None if ENVIRONMENT == "local" else (
                os.getenv("OPENSEARCH_USER"),
                os.getenv("OPENSEARCH_PASSWORD")
            ),
            use_ssl=ENVIRONMENT != "local",
            verify_certs=ENVIRONMENT != "local",
            connection_class=RequestsHttpConnection
        )

        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # Apply same re-ranking logic
        query_keywords = set(extract_keywords(query, top_n=5))
        
        ranked_results = []
        for doc, vector_score in results:
            doc_keywords = set(extract_keywords(doc.page_content, top_n=10))
            keyword_matches = len(query_keywords.intersection(doc_keywords))
            
            query_words = query.lower().split()
            phrase_matches = sum(1 for word in query_words if word in doc.page_content.lower())
            
            combined_score = vector_score - (keyword_matches * 0.05) - (phrase_matches * 0.02)
            
            ranked_results.append({
                'doc': doc,
                'vector_score': vector_score,
                'keyword_matches': keyword_matches,
                'phrase_matches': phrase_matches,
                'combined_score': combined_score
            })
        
        ranked_results.sort(key=lambda x: x['combined_score'])

        return jsonify({
            "query": query,
            "index": index_name,
            "query_keywords": list(query_keywords),
            "num_results": len(ranked_results),
            "results": [
                {
                    "rank": i + 1,
                    "vector_score": r['vector_score'],
                    "keyword_matches": r['keyword_matches'],
                    "phrase_matches": r['phrase_matches'],
                    "combined_score": r['combined_score'],
                    "content_preview": r['doc'].page_content[:300] + "...",
                    "full_length": len(r['doc'].page_content),
                    "metadata": r['doc'].metadata
                }
                for i, r in enumerate(ranked_results)
            ]
        }), 200

    except Exception as e:
        logger.error("Debug search failed", exc_info=True)
        return jsonify({"error": str(e)}), 500

# =============================================================================
# LIST INDICES
# =============================================================================

@app.route("/list-indices", methods=["GET"])
def list_indices():
    """List all available indices"""
    try:
        indices = opensearch_client.indices.get_alias(index="*")
        
        index_info = []
        for index_name in indices.keys():
            if not index_name.startswith('.'):
                stats = opensearch_client.indices.stats(index=index_name)
                index_info.append({
                    "name": index_name,
                    "document_count": stats['indices'][index_name]['total']['docs']['count'],
                    "size_bytes": stats['indices'][index_name]['total']['store']['size_in_bytes']
                })
        
        return jsonify({
            "indices": index_info,
            "total_count": len(index_info)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing indices: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =============================================================================
# DELETE INDEX
# =============================================================================

@app.route("/delete-index", methods=["DELETE"])
def delete_index():
    """Delete an index"""
    try:
        data = request.json or {}
        index_name = data.get("store_id", DEFAULT_INDEX).lower().replace(" ", "_")
        
        if opensearch_client.indices.exists(index=index_name):
            opensearch_client.indices.delete(index=index_name)
            logger.info(f"Index '{index_name}' deleted successfully")
            return jsonify({
                "status": "success",
                "message": f"Index '{index_name}' deleted successfully"
            }), 200
        else:
            return jsonify({
                "error": f"Index '{index_name}' does not exist"
            }), 404
            
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8770))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting RAG backend on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)