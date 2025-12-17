from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
from bs4 import BeautifulSoup

from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
INDEX_NAME = "rag_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

# --------------------------------------------------
# APP
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# --------------------------------------------------
# OPENSEARCH
# --------------------------------------------------

opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection
)

opensearch_url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def fetch_text(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract text from p, div, article
        texts = []
        for tag in soup.find_all(["p", "div", "article"]):
            t = tag.get_text(strip=True)
            if len(t) > 50:
                texts.append(t)

        text = "\n\n".join(texts)
        return text if len(text) > 300 else None
    except Exception as e:
        logger.error(f"Fetch failed: {url} → {e}")
        return None




def get_vectorstore(index_name: str):
    return OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embedding_model,
        opensearch_url=opensearch_url
    )

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "opensearch": opensearch_client.cluster.health()
    })


@app.route("/process-urls", methods=["POST"])
def process_urls():
    """
    Fetch URLs, extract full article text, chunk, and index into OpenSearch.
    """
    data = request.json or {}
    urls = data.get("urls", [])
    index_name = data.get("store_id", INDEX_NAME)

    if not urls:
        return jsonify({"error": "urls required"}), 400

    docs = []

    for url in urls:
        text = fetch_text(url)
        if not text:
            logger.warning(f"No content extracted from {url}")
            continue

        logger.info(f"Fetched {len(text)} chars from {url}")
        docs.append(Document(
            page_content=text,
            metadata={"source": url}
        ))

    if not docs:
        return jsonify({"error": "No content extracted from any URL"}), 400

    # Split into smaller chunks with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,    # smaller chunks capture more context
        chunk_overlap=200  # overlap to keep continuity
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks for indexing")

    # Index into OpenSearch
    OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
        opensearch_url=opensearch_url
    )

    logger.info(f"Indexed {len(chunks)} chunks into OpenSearch index '{index_name}'")

    return jsonify({
        "status": "success",
        "index": index_name,
        "chunks": len(chunks)
    })


@app.route("/rag/ask", methods=["POST"])
def ask():
    data = request.json or {}
    query = data.get("query")
    k = int(data.get("k", 10))  # Increase k to 10 or more for better results
    index_name = data.get("store_id", INDEX_NAME)

    if not query:
        return jsonify({"error": "query required"}), 400

    # Get the vector store for the specific index
    vectorstore = get_vectorstore(index_name)

    # Perform similarity search with increased k (top 10 results)
    results = vectorstore.similarity_search(query, k=k)

    # Get the context from the results and the sources
    context = "\n\n".join(d.page_content for d in results)
    sources = list({d.metadata["source"] for d in results})

    return jsonify({
        "query": query,
        "answer": context[:2000],  # Return up to 2000 characters
        "sources": sources
    })


@app.route("/delete-index", methods=["DELETE"])
def delete_index():
    data = request.json or {}
    index_name = data.get("store_id", INDEX_NAME)

    if opensearch_client.indices.exists(index=index_name):
        opensearch_client.indices.delete(index=index_name)
        return jsonify({"status": "deleted"})

    return jsonify({"error": "index not found"}), 404


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8770)
