import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# Load API Key
# -----------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# Load PDF
# -----------------------------

pdf_path = "sample.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"'{pdf_path}' not found in current directory. Please place a PDF file named '{pdf_path}' in the project root directory: {os.getcwd()}"
    )

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Pages Loaded:", len(documents))

# -----------------------------
# Split into Chunks
# -----------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=50

)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

# -----------------------------
# Create Embeddings
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(

    model="models/embedding-001",

    google_api_key=api_key

)

# -----------------------------
# Store in ChromaDB
# -----------------------------

db = Chroma.from_documents(

    documents=chunks,

    embedding=embeddings

)

print("Vector Database Created")

# -----------------------------
# Create Retriever
# -----------------------------

retriever = db.as_retriever(

    search_kwargs={"k":3}

)

# -----------------------------
# Load LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(

    model="gemini-2.5-flash",

    google_api_key=api_key

)

# -----------------------------
# Ask Question
# -----------------------------

question = input("Ask Question: ")

# -----------------------------
# Retrieve Documents
# -----------------------------

docs = retriever.invoke(question)

print("\nRetrieved Chunks\n")

for doc in docs:

    print(doc.page_content)

    print("-"*50)

# -----------------------------
# Build Prompt
# -----------------------------

context = ""

for doc in docs:

    context += doc.page_content + "\n"

prompt = f"""

Answer only using the context below.

Context:

{context}

Question:

{question}

"""

# -----------------------------
# Generate Answer
# -----------------------------

response = llm.invoke(prompt)

print("\nAnswer\n")

print(response.content)