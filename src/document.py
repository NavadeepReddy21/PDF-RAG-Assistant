from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from . import config

def process_pdf(pdf_path: str):
    """Loads a PDF and splits it into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    
    return documents, chunks
