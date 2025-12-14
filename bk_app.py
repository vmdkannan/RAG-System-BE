from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.schema import Document
import os
import logging
from bs4 import BeautifulSoup
import requests
from datetime import datetime 
from rag_utils import (
    prepare_context,
    construct_prompt,
    generate_with_huggingface,
    process_response,
    HuggingFaceLLM,
    get_available_models
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# GLOBAL LLM INSTANCE (FIXED - Now properly used)
# ============================================================================
llm_instance = None
llm_lock = False  # Simple lock to prevent concurrent loading

def get_llm_instance(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """Get or create LLM instance (singleton pattern)"""
    global llm_instance, llm_lock
    
    # If model is already loaded and matches requested model
    if llm_instance is not None and llm_instance.model_name == model_name:
        logger.info(f"Using cached LLM instance: {model_name}")
        return llm_instance
    
    # If different model requested, reload
    if llm_instance is not None and llm_instance.model_name != model_name:
        logger.info(f"Switching from {llm_instance.model_name} to {model_name}")
        llm_instance = None
    
    # Load new model
    if llm_instance is None and not llm_lock:
        try:
            llm_lock = True
            logger.info(f"Loading new LLM instance: {model_name}")
            llm_instance = HuggingFaceLLM(model_name=model_name)
            llm_instance.load_model()
            llm_lock = False
            logger.info("LLM instance loaded successfully")
        except Exception as e:
            llm_lock = False
            logger.error(f"Failed to load LLM: {str(e)}")
            raise
    
    return llm_instance

# ============================================
# Initialize Embedding Model
# ============================================
logger.info("Loading embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    raise

# ============================================
# OpenSearch Configuration
# ============================================
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

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
            http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASSWORD")),
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

# ============================================
# Helper function to clean web content
# ============================================
def clean_web_content(url):
    """Extract clean content from URL using BeautifulSoup"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()

        main_content = (
            soup.find('article') or 
            soup.find('main') or 
            soup.find('div', class_=['content', 'article', 'post']) or
            soup.find('body')
        )

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
            text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
            return text

        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return None

# ============================================
# Endpoints
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        opensearch_health = opensearch_client.cluster.health()
        return jsonify({
            "status": "healthy",
            "opensearch": {
                "status": opensearch_health.get("status"),
                "cluster_name": opensearch_health.get("cluster_name"),
                "number_of_nodes": opensearch_health.get("number_of_nodes")
            },
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_loaded": llm_instance is not None and llm_instance.initialized if llm_instance else False
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/process-urls', methods=['POST'])
def process_urls():
    """Process URLs and create vector store in OpenSearch"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        urls = data.get('urls', [])
        index_name = data.get('store_id', 'default_index').lower().replace(' ', '_')
        chunk_size = data.get('chunk_size', 1000)
        chunk_overlap = data.get('chunk_overlap', 200)

        if not urls or not isinstance(urls, list):
            return jsonify({"error": "Valid URLs list required"}), 400

        documents = []
        
        for url in urls:
            clean_text = clean_web_content(url)
            if clean_text and len(clean_text) > 100:
                doc = Document(page_content=clean_text, metadata={"source": url, "type": "web_article"})
                documents.append(doc)

        if not documents:
            return jsonify({"error": "No valid content could be extracted from URLs"}), 400

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        quality_chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

        if not quality_chunks:
            return jsonify({"error": "No quality content chunks created"}), 400

        # Create vector store
        vectorstore = OpenSearchVectorSearch.from_documents(
            documents=quality_chunks,
            embedding=embedding_model,
            opensearch_url=opensearch_url,
            index_name=index_name,
            http_auth=None if ENVIRONMENT == "local" else (os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASSWORD")),
            use_ssl=False if ENVIRONMENT == "local" else True,
            verify_certs=False if ENVIRONMENT == "local" else True,
            connection_class=RequestsHttpConnection,
            engine="faiss",
            space_type="l2",
            bulk_size=500
        )

        return jsonify({
            "status": "success",
            "message": f"Processed {len(urls)} URLs into {len(quality_chunks)} chunks",
            "index_name": index_name,
            "num_documents": len(documents),
            "num_chunks": len(quality_chunks)
        }), 200

    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to process URLs", "details": str(e)}), 500

# ============================================================================
# RAG ENDPOINT - Complete Pipeline (OPTIMIZED)
# ============================================================================

@app.route('/rag/ask', methods=['POST'])
def rag_ask():
    """
    Complete RAG pipeline endpoint - OPTIMIZED VERSION
    Uses cached LLM instance for better performance
    """
    try:
        data = request.json
        
        # Extract parameters
        query = data.get('query')
        index_name = data.get('store_id', 'default_index').lower().replace(' ', '_')
        k = data.get('k', 4)
        score_threshold = data.get('score_threshold', 0.3)
        max_context_length = data.get('max_context_length', 4000)
        prompt_template = data.get('prompt_template', 'default')
        model_name = data.get('model_name', 'mistralai/Mistral-7B-Instruct-v0.2')
        max_new_tokens = data.get('max_new_tokens', 512)
        temperature = data.get('temperature', 0.7)
        include_citations = data.get('include_citations', True)
        format_markdown = data.get('format_markdown', True)
        
        # Validate query
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Check if index exists
        if not opensearch_client.indices.exists(index=index_name):
            all_indices = opensearch_client.indices.get_alias(index="*")
            available_stores = [i for i in all_indices.keys() if not i.startswith('.')]
            return jsonify({
                "error": f"Vector store '{index_name}' not found",
                "available_stores": available_stores
            }), 404
        
        # Step 1: Query vector store
        logger.info(f"Querying vector store: {index_name}")
        vectorstore = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding_model,
            http_auth=None if ENVIRONMENT == "local" else (os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASSWORD")),
            use_ssl=False if ENVIRONMENT == "local" else True,
            verify_certs=False if ENVIRONMENT == "local" else True,
            connection_class=RequestsHttpConnection
        )
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        # Step 2: Prepare context
        logger.info("Preparing context...")
        context_data = prepare_context(
            results=results,
            score_threshold=score_threshold,
            max_context_length=max_context_length,
            include_metadata=True
        )
        
        # Check if context is empty
        if context_data['is_empty']:
            return jsonify({
                "answer": "I couldn't find relevant information in the knowledge base to answer your question.",
                "query": query,
                "sources": [],
                "context_info": context_data,
                "message": "No relevant documents found"
            }), 200
        
        # Step 3: Construct prompt
        logger.info("Constructing prompt...")
        prompt = construct_prompt(
            query=query,
            context=context_data['context'],
            prompt_template=prompt_template
        )
        
        # Step 4: Generate response with HuggingFace (OPTIMIZED - uses cached instance)
        logger.info(f"Generating response with model: {model_name}")
        start_time = datetime.now()
        
        try:
            # Use cached LLM instance
            llm = get_llm_instance(model_name=model_name)
            response = llm.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            generation_result = {
                "response": response,
                "model": model_name,
                "success": True,
                "error": None,
                "generation_time": generation_time
            }
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            generation_result = {
                "response": "",
                "model": model_name,
                "success": False,
                "error": str(e),
                "generation_time": 0.0
            }
        
        # Check if generation was successful
        if not generation_result['success']:
            return jsonify({
                "error": "Failed to generate response",
                "details": generation_result['error'],
                "query": query,
                "sources": context_data['sources']
            }), 500
        
        # Step 5: Process response
        logger.info("Processing response...")
        final_response = process_response(
            raw_response=generation_result['response'],
            sources=context_data['sources'],
            query=query,
            include_citations=include_citations,
            format_markdown=format_markdown
        )
        
        # Add additional metadata
        final_response['model'] = model_name
        final_response['generation_time'] = generation_result['generation_time']
        final_response['context_info'] = {
            'num_sources_retrieved': context_data['num_sources'],
            'total_characters': context_data['total_characters'],
            'filtered_count': context_data['filtered_count']
        }
        
        return jsonify(final_response), 200
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to process RAG request",
            "details": str(e)
        }), 500

# ============================================================================
# ADDITIONAL RAG ENDPOINTS (For testing and debugging)
# ============================================================================

@app.route('/rag/context', methods=['POST'])
def rag_get_context():
    """Get context from vector store only (for testing)"""
    try:
        data = request.json
        query = data.get('query')
        index_name = data.get('store_id', 'default_index').lower().replace(' ', '_')
        k = data.get('k', 4)
        score_threshold = data.get('score_threshold', 0.3)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        if not opensearch_client.indices.exists(index=index_name):
            return jsonify({"error": f"Vector store '{index_name}' not found"}), 404
        
        vectorstore = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name=index_name,
            embedding_function=embedding_model,
            http_auth=None if ENVIRONMENT == "local" else (os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASSWORD")),
            use_ssl=False if ENVIRONMENT == "local" else True,
            verify_certs=False if ENVIRONMENT == "local" else True,
            connection_class=RequestsHttpConnection
        )
        
        results = vectorstore.similarity_search_with_score(query, k=k)
        context_data = prepare_context(
            results=results,
            score_threshold=score_threshold,
            include_metadata=True
        )
        
        return jsonify({
            "query": query,
            "context_data": context_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/rag/models', methods=['GET'])
def rag_available_models():
    """Get list of recommended HuggingFace models"""
    try:
        models = get_available_models()
        return jsonify({
            "models": models,
            "recommended": "mistralai/Mistral-7B-Instruct-v0.2",
            "currently_loaded": llm_instance.model_name if llm_instance else None,
            "note": "Switching models will reload the LLM (may take time)"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rag/health', methods=['GET'])
def rag_health():
    """Check if RAG pipeline is ready"""
    try:
        import torch
        
        status = {
            "rag_utils_loaded": True,
            "torch_available": torch.cuda.is_available(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "llm_loaded": llm_instance is not None and llm_instance.initialized if llm_instance else False,
            "llm_model": llm_instance.model_name if llm_instance and llm_instance.initialized else None,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        return jsonify(status), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "rag_utils_loaded": False
        }), 500

@app.route('/list-indices', methods=['GET'])
def list_indices():
    """List all vector store indices"""
    try:
        indices = opensearch_client.indices.get_alias(index="*")
        index_info = []
        for idx in indices.keys():
            if not idx.startswith('.'):
                stats = opensearch_client.indices.stats(index=idx)
                index_info.append({
                    "name": idx,
                    "document_count": stats['indices'][idx]['total']['docs']['count'],
                    "size": stats['indices'][idx]['total']['store']['size_in_bytes']
                })
        return jsonify({"indices": index_info, "total_count": len(index_info)}), 200
    except Exception as e:
        logger.error(f"Error listing indices: {str(e)}")
        return jsonify({"error": "Failed to list indices", "details": str(e)}), 500

@app.route('/delete-index', methods=['DELETE'])
def delete_index():
    """Delete a vector store index"""
    try:
        data = request.json
        index_name = data.get('store_id', 'default_index').lower().replace(' ', '_')
        if opensearch_client.indices.exists(index=index_name):
            opensearch_client.indices.delete(index=index_name)
            return jsonify({"status": "success", "message": f"Index '{index_name}' deleted successfully"}), 200
        return jsonify({"error": f"Index '{index_name}' does not exist"}), 404
    except Exception as e:
        logger.error(f"Error deleting index: {str(e)}")
        return jsonify({"error": "Failed to delete index", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8770))
    app.run(debug=os.getenv('DEBUG', 'false').lower() == 'true', host='0.0.0.0', port=port)