import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()
apikey = os.getenv("GOOGLE_API_KEY")

pdf_path = "sample.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"'{pdf_path}' not found in current directory. Please place a PDF named '{pdf_path}' in the project root directory: {os.getcwd()}"
    )

documents = PyPDFLoader(pdf_path).load()
print("Pages Loaded:", len(documents))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=apikey)
db=Chroma.from_documents(chunks, embeddings)
print("db created with", len(chunks), "chunks")

retriever=db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="models/chat-bison-001", google_api_key=apikey)

question=input("Ask a question: ")
docs=retriever.invoke(question)

for doc in docs:
    print("Document:", doc.page_content)

context= ""
for doc in docs:
    context += doc.page_content + "\n"

prompt=f"""You are a helpful assistant. Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""

result=llm.invoke(prompt)
print("Answer:", result)
