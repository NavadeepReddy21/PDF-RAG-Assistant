from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from . import config

def create_vector_store(chunks):
    """Creates a ChromaDB vector store from document chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL, 
        google_api_key=config.GOOGLE_API_KEY
    )
    
    db = Chroma.from_documents(chunks, embeddings)
    return db

def get_retriever(db):
    """Returns a retriever for the vector store."""
    return db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
