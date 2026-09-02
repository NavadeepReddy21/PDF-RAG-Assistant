# PDF-RAG-Assistant

A Retrieval-Augmented Generation (RAG) chatbot that enables users to interact with PDF documents through natural language queries. The application combines semantic search with Google's Gemini model to provide accurate, context-aware answers based on the content of a PDF.

---

## Features

* PDF document ingestion and processing
* Intelligent text chunking with overlap
* Semantic embeddings using Sentence Transformers
* Vector storage with ChromaDB
* Context-aware retrieval
* AI-powered question answering with Gemini
* Source page tracking for answer transparency

---

## Architecture

```text
PDF → Text Extraction → Chunking → Embeddings → ChromaDB
                                                    │
                                                    ▼
User Query → Semantic Retrieval → Relevant Chunks → Gemini → Answer
```

---

## Tech Stack

| Technology            | Purpose                 |
| --------------------- | ----------------------- |
| Python                | Core application        |
| LangChain             | RAG orchestration       |
| ChromaDB              | Vector database         |
| Sentence Transformers | Embedding generation    |
| Google Gemini         | Large Language Model    |
| PyPDF                 | PDF loading and parsing |

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PDF_CHATBOT.git
cd PDF_CHATBOT
```

### Create and Activate a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

---

## Usage

1. Place your PDF file in the project directory.
2. Rename the file to:

```text
document.pdf
```

3. Run the application:

```bash
python app.py
```

4. Start asking questions about the document.

### Example

```text
You: What is the purpose of this document?

Answer: This document explains...

Source pages: {1, 2}
```

---

## Project Structure

```text
PDF_CHATBOT/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── document.pdf
│
├── chroma_db/        # Generated vector database
├── venv/             # Virtual environment
└── .env              # API key configuration
```

---

## How It Works

1. The PDF is loaded and parsed.
2. Text is split into smaller overlapping chunks.
3. Each chunk is converted into a vector embedding.
4. Embeddings are stored in ChromaDB.
5. User queries are converted into embeddings.
6. Relevant document chunks are retrieved using semantic similarity search.
7. Gemini generates an answer using the retrieved context.
8. Source pages are displayed alongside the response.

---

## Future Enhancements

* Web interface with Streamlit
* Multi-PDF support
* Conversational memory
* Source citation highlighting
* Docker containerization
* Cloud deployment
* PDF upload functionality

---

## Sample Output

```text
PDF Chatbot ready! Ask anything about your document.

You: Summarize the document.

Answer:
The document discusses...

Source pages: {0, 1, 2}
```

---

## License

This project is licensed under the MIT License.

---

## Author

Developed as a hands-on project to explore Retrieval-Augmented Generation (RAG), vector databases, semantic search, and Large Language Models.
