import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path ="./chroma_db"
)

collection = client.get_or_create_collection(
    name="test_docs",
    embedding_function=embedding_function
)

documents = [
    "ChromaDB is an open-source vector database.",
    "Vector databases store embeddings for semantic search.",
    "Embeddings convert text into numerical vectors.",
    "ChromaDB is commonly used in RAG applications."
]

ids = ["doc1", "doc2", "doc3", "doc4"]

collection.add(
    documents=documents,
    ids=ids
)

# Query
query = "php is good programming lanaguage"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print(results["documents"])

print("DB persisted")
import time
time.sleep(2)

