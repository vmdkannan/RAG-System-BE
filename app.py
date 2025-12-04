from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize embedding model once (loaded at startup)
logger.info("Initializing embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("Embedding model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {e}")
    raise

# Store vector stores in memory
vector_stores = {}

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        "message": "RAG Backend API",
        "version": "1.0",
        "endpoints": {
            "GET /health": "Health check",
            "POST /process-urls": "Process URLs and create vector store",
            "POST /query": "Query the vector store",
            "GET /stores": "List all vector stores",
            "DELETE /stores/<store_id>": "Delete a vector store"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "active_stores": len(vector_stores)
    }), 200

@app.route('/process-urls', methods=['POST'])
def process_urls():
    """
    Process URLs and create vector store
    Expected JSON body:
    {
        "urls": ["url1", "url2", ...],
        "store_id": "optional_identifier",
        "chunk_size": 500,  # optional
        "chunk_overlap": 50  # optional
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        urls = data.get('urls', [])
        store_id = data.get('store_id', 'default')
        chunk_size = data.get('chunk_size', 500)
        chunk_overlap = data.get('chunk_overlap', 50)
        
        if not urls:
            return jsonify({"error": "No URLs provided"}), 400
        
        if not isinstance(urls, list):
            return jsonify({"error": "URLs must be provided as a list"}), 400
        
        logger.info(f"Processing {len(urls)} URLs for store_id: {store_id}")
        
        # 1. Load documents
        logger.info("Loading documents from URLs...")
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            return jsonify({"error": "No documents could be loaded from the provided URLs"}), 400
        
        # 2. Split documents
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # 3. Create vector store
        logger.info("Creating vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        
        # Store in memory
        vector_stores[store_id] = vectorstore
        logger.info(f"Vector store '{store_id}' created successfully")
        
        return jsonify({
            "status": "success",
            "message": f"Processed {len(urls)} URLs into {len(chunks)} chunks",
            "store_id": store_id,
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}")
        return jsonify({
            "error": "Failed to process URLs",
            "details": str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query():
    """
    Query the vector store
    Expected JSON body:
    {
        "query": "your question",
        "store_id": "optional_identifier",
        "k": 3,  # number of results to return
        "score_threshold": 0.0  # optional minimum similarity score
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query_text = data.get('query')
        store_id = data.get('store_id', 'default')
        k = data.get('k', 3)
        score_threshold = data.get('score_threshold', 0.0)
        
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        if store_id not in vector_stores:
            return jsonify({
                "error": f"Vector store '{store_id}' not found",
                "available_stores": list(vector_stores.keys())
            }), 404
        
        logger.info(f"Querying store '{store_id}' with: {query_text}")
        
        # Perform similarity search
        vectorstore = vector_stores[store_id]
        
        if score_threshold > 0:
            results = vectorstore.similarity_search_with_score(query_text, k=k)
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= score_threshold
            ]
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in filtered_results
            ]
        else:
            results = vectorstore.similarity_search(query_text, k=k)
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        
        logger.info(f"Found {len(formatted_results)} results")
        
        return jsonify({
            "status": "success",
            "query": query_text,
            "store_id": store_id,
            "num_results": len(formatted_results),
            "results": formatted_results
        }), 200
        
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        return jsonify({
            "error": "Failed to query vector store",
            "details": str(e)
        }), 500

@app.route('/stores', methods=['GET'])
def list_stores():
    """List all available vector stores"""
    store_info = {}
    for store_id, vectorstore in vector_stores.items():
        try:
            # Try to get the index size
            store_info[store_id] = {
                "exists": True,
                "type": "FAISS"
            }
        except:
            store_info[store_id] = {
                "exists": True,
                "type": "Unknown"
            }
    
    return jsonify({
        "status": "success",
        "count": len(vector_stores),
        "stores": store_info
    }), 200

@app.route('/stores/<store_id>', methods=['DELETE'])
def delete_store(store_id):
    """Delete a vector store"""
    if store_id in vector_stores:
        del vector_stores[store_id]
        logger.info(f"Deleted vector store: {store_id}")
        return jsonify({
            "status": "success",
            "message": f"Store '{store_id}' deleted"
        }), 200
    else:
        return jsonify({
            "error": f"Store '{store_id}' not found",
            "available_stores": list(vector_stores.keys())
        }), 404

@app.route('/stores/<store_id>/info', methods=['GET'])
def store_info(store_id):
    """Get information about a specific vector store"""
    if store_id not in vector_stores:
        return jsonify({
            "error": f"Store '{store_id}' not found",
            "available_stores": list(vector_stores.keys())
        }), 404
    
    try:
        vectorstore = vector_stores[store_id]
        return jsonify({
            "status": "success",
            "store_id": store_id,
            "type": "FAISS",
            "exists": True
        }), 200
    except Exception as e:
        logger.error(f"Error getting store info: {str(e)}")
        return jsonify({
            "error": "Failed to get store information",
            "details": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8770))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting RAG Backend API on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)