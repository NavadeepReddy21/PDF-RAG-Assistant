import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Models
EMBEDDING_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval Configuration
RETRIEVER_K = 3
