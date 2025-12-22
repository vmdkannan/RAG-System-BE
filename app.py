from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
from bs4 import BeautifulSoup
import tempfile

from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders.pdf import PyPDFLoader


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))
INDEX_NAME = "rag_index"

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  


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
# OPENSEARCH CLIENT
# --------------------------------------------------

opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection
)

opensearch_url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"

# --------------------------------------------------
# DIRECT HUGGINGFACE API CALL
# --------------------------------------------------

def call_huggingface_api(prompt: str, max_tokens: int = 256) -> str:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")

    client = InferenceClient(api_key=HF_TOKEN)

    try:
        logger.info(f"Calling HuggingFace chat model: {HF_MODEL}")

        completion = client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        logger.exception("Error calling HuggingFace API")
        raise RuntimeError(f"HuggingFace API error: {e}")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def fetch_text(url: str) -> str | None:
    """Fetch main textual content from a URL"""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 50
        ]

        text = "\n\n".join(paragraphs)
        return text if len(text) > 300 else None

    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {e}")
        return None

def get_vectorstore(index_name: str):
    """Return OpenSearchVectorSearch instance"""
    return OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embedding_model,
        opensearch_url=opensearch_url
    )
    
def fetch_pdf(file):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} pages")
        return documents

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "opensearch": opensearch_client.cluster.health(),
        "llm_configured": HF_TOKEN is not None,
        "llm_model": HF_MODEL
    })

@app.route("/process-urls", methods=["POST"])
def process_urls():
    data = request.json or {}
    urls = data.get("urls", [])
    index_name = data.get("store_id", INDEX_NAME)

    if not urls:
        return jsonify({"error": "urls required"}), 400

    docs = []
    for url in urls:
        text = fetch_text(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url}))

    if not docs:
        return jsonify({"error": "No content extracted"}), 400

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
        opensearch_url=opensearch_url
    )

    return jsonify({
        "status": "success",
        "index": index_name,
        "chunks_indexed": len(chunks)
    })
    

@app.route("/process-pdf", methods=["POST"])
def process_pdf():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Optional index name from form-data
    index_name = request.form.get("store_id", INDEX_NAME)

    try:
        documents = fetch_pdf(file)

        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        OpenSearchVectorSearch.from_documents(
            documents=chunks,
            embedding=embedding_model,
            index_name=index_name,
            opensearch_url=opensearch_url,
            bulk_size=1500
        )

        return jsonify({
            "status": "success",
            "index": index_name,
            "chunks_indexed": len(chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    

@app.route("/rag/ask", methods=["POST"])
def ask():
    try:
        data = request.json or {}
        query = data.get("query")
        k = int(data.get("k", 5))
        index_name = data.get("store_id", INDEX_NAME)

        if not query:
            return jsonify({"error": "query required"}), 400

        if not HF_TOKEN:
            return jsonify({"error": "HUGGINGFACE_API_TOKEN not configured"}), 500

        # Get vectorstore and retrieve documents
        vectorstore = get_vectorstore(index_name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)

        if not docs:
            return jsonify({
                "query": query,
                "answer": "No relevant information found.",
                "sources": []
            })

        # Format context and sources
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = list({d.metadata["source"] for d in docs})

        # Create prompt - OPTIMIZED FOR FLAN-T5
        prompt = f"""Answer this question using only the context provided.

        Context: {context}

        Question: {query}

        Answer:"""

        logger.info(f"Generating answer for: {query}")
        answer = call_huggingface_api(prompt, max_tokens=256)

        return jsonify({
            "query": query,
            "answer": answer,
            "sources": sources,
            "num_sources": len(docs)
        })

    except Exception as e:
        logger.error(f"Error in /rag/ask: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to generate answer",
            "details": str(e)
        }), 500

@app.route("/delete-index", methods=["DELETE"])
def delete_index():
    data = request.json or {}
    index_name = data.get("store_id", INDEX_NAME)

    if opensearch_client.indices.exists(index=index_name):
        opensearch_client.indices.delete(index=index_name)
        return jsonify({"status": "deleted"})

    return jsonify({"error": "index not found"}), 404

@app.route("/test-model", methods=["GET"])
def test_model():
    """Test if current model is working"""
    try:
        test_prompt = "What is 2+2? Answer:"
        result = call_huggingface_api(test_prompt, max_tokens=50)
        return jsonify({
            "status": "working",
            "model": HF_MODEL,
            "test_response": result
        })
    except Exception as e:
        return jsonify({
            "status": "failed",
            "model": HF_MODEL,
            "error": str(e)
        }), 500

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8770, debug=False)