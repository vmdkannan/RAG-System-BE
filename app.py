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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

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
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
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

@app.route('/query', methods=['POST'])
def query_vectorstore():
    """Query the vector store"""
    try:
        data = request.json
        query = data.get('query')
        index_name = data.get('store_id', 'default_index').lower().replace(' ', '_')
        k = data.get('k', 4)

        if not query:
            return jsonify({"error": "No query provided"}), 400

        if not opensearch_client.indices.exists(index=index_name):
            all_indices = opensearch_client.indices.get_alias(index="*")
            available_stores = [i for i in all_indices.keys() if not i.startswith('.')]
            return jsonify({
                "error": f"Vector store '{index_name}' not found",
                "available_stores": available_stores
            }), 404

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
        return jsonify({
            "query": query,
            "index_name": index_name,
            "results": [{"content": d.page_content, "metadata": d.metadata, "score": float(s)} for d, s in results],
            "num_results": len(results)
        }), 200

    except Exception as e:
        logger.error(f"Error querying: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to query vector store", "details": str(e)}), 500

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
