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
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import lcdbaccess as dbaccess


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

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

# Load a pre-trained Hugging Face model and tokenizer
model_name = "t5-small"  # You can use other models like 'facebook/bart-large-cnn' for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Initialize Hugging Face summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# --------------------------------------------------
# LLM SUMMARIZER
# --------------------------------------------------

summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1  # CPU
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def fetch_text(url: str) -> str | None:
    """Basic but reliable text extraction with logging for debugging"""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 50
        ]

        text = "\n\n".join(paragraphs)

        # Log for manual inspection
        if text:
            logger.info(f"✅ Extracted {len(paragraphs)} paragraphs from {url}")
            logger.debug(f"Extracted text:\n{text[:2000]}...")  # log first 2000 chars
        else:
            logger.warning(f"⚠️ No sufficient content extracted from {url}")

        return text if len(text) > 300 else None

    except Exception as e:
        logger.error(f"Fetch failed: {url} → {e}")
        return None


def get_vectorstore(index_name: str):
    """Return OpenSearchVectorSearch instance"""
    return OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embedding_model,
        opensearch_url=dbaccess.opensearch_url
    )

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "opensearch": dbaccess.opensearch_client.cluster.health()
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
        if not text:
            continue

        # Split into paragraphs
        paragraphs = text.split("\n\n")
        filtered_paragraphs = [
            p for p in paragraphs
            if len(p) > 50 and not any(x in p.lower() for x in ["cookie", "privacy policy", "i agree", "click 'i accept'"])
        ]

        for p in filtered_paragraphs:
            docs.append(Document(page_content=p, metadata={"source": url}))

    # Smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i} ({len(chunk.page_content)} chars): {chunk.page_content[:500]}...")


    # Index the chunks in OpenSearch
    OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
        opensearch_url=dbaccess.opensearch_url
    )

    return jsonify({
        "status": "success",
        "index": index_name,
        "chunks_indexed": len(chunks)
    })


@app.route("/rag/ask", methods=["POST"])
def ask():
    data = request.json or {}
    query = data.get("query")
    k = int(data.get("k", 5))
    index_name = data.get("store_id", INDEX_NAME)

    if not query:
        return jsonify({"error": "query required"}), 400

    # Fetch the vectorstore and perform similarity search
    vectorstore = get_vectorstore(index_name)
    results = vectorstore.similarity_search(query, k=k)

    context = "\n\n".join(d.page_content for d in results)
    sources = list({d.metadata["source"] for d in results})

    # Prepare the input for the Hugging Face model
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"

    # Use Hugging Face LLM to generate an answer based on the context
    # The output of the summarizer pipeline is a list of dictionaries
    response = summarizer(input_text, max_length=200, min_length=50, do_sample=False)
    
    # The output is a list, so extract the generated text correctly
    answer = response[0]['generated_text']  # Use 'generated_text' for text generation models

    return jsonify({
        "query": query,
        "answer": answer,
        "sources": sources
    })


@app.route("/delete-index", methods=["DELETE"])
def delete_index():
    data = request.json or {}
    index_name = data.get("store_id", INDEX_NAME)

    if dbaccess.opensearch_client.indices.exists(index=index_name):
        dbaccess.opensearch_client.indices.delete(index=index_name)
        return jsonify({"status": "deleted"})

    return jsonify({"error": "index not found"}), 404

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8770)
