import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Embedding function (official Chroma wrapper)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    normalize_embeddings=True
)

# Initialize Chroma client (auto-persistent)
client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db"
    )
)

# Create or load collection
collection = client.get_or_create_collection(
    name="company_docs",
    embedding_function=embedding_function
)

# Documents
documents = [
    "ChromaDB is an open-source vector database.",
    "Vector databases store embeddings for semantic search.",
    "Embeddings convert text into numerical vectors.",
    "ChromaDB is commonly used in RAG applications."
]

ids = ["doc1", "doc2", "doc3", "doc4"]

# Add documents
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
