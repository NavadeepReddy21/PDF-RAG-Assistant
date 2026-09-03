import uuid
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from . import config

def create_vector_store(chunks):
    """Creates a ChromaDB vector store from document chunks."""
    # Using local embeddings to avoid Google API rate limits
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Use a unique collection name to avoid dimension mismatch with old embeddings in memory
    collection_name = f"pdf_{uuid.uuid4().hex}"
    db = Chroma.from_documents(chunks, embeddings, collection_name=collection_name)
    return db

def get_retriever(db):
    """Returns a retriever for the vector store."""
    return db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
