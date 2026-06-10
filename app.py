import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# ── Step 1: Load PDF ──
print("Loading PDF...")
loader = PyPDFLoader("document.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# ── Step 2: Split into chunks ──
# LLMs can't read 100 pages at once
# So we cut into small overlapping chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk = 500 characters
    chunk_overlap=50      # 50 characters overlap between chunks
                          # overlap ensures no context is lost at boundaries
)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

# ── Step 3: Convert chunks to embeddings + store ──
print("Creating embeddings... (takes 1-2 minutes first time)")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # saves to disk
)
print("Embeddings stored!\n")

# ── Step 4: Create retriever ──
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}   # fetch top 3 relevant chunks
)

# ── Step 5: Connect Gemini ──
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3    # low = more factual answers
)

# ── Step 6: Create RAG chain ──
# This connects retriever + llm automatically
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True   # shows which chunks were used
)

# ── Step 7: Chat loop ──
print("PDF Chatbot ready! Ask anything about your document.")
print("Type 'quit' to exit\n")

while True:
    question = input("You: ")
    
    if question.lower() == "quit":
        print("Bye!")
        break
    
    result = qa_chain.invoke({"query": question})
    
    print(f"\nAnswer: {result['result']}")
    
    # Show which page the answer came from
    sources = result["source_documents"]
    pages_used = set([doc.metadata.get("page", "?") for doc in sources])
    print(f"Source pages: {pages_used}\n")