from flask import Flask, request, jsonify
import os

from opensearchpy import OpenSearch, RequestsHttpConnection

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", 9200))


opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection
)


opensearch_url = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"