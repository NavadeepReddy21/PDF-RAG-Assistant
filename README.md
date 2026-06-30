# 📄 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF
and ask questions about it in natural language. Combines semantic search
with Google's Gemini model for accurate, context-aware answers.

🔗 **Live demo:** [Add your Hugging Face Space link here after deployment]

## Features
- Upload any PDF directly in the browser (no need to rename files or place them in a folder)
- Intelligent text chunking with overlap for better context retention
- Semantic embeddings via Sentence Transformers (`all-MiniLM-L6-v2`)
- Vector storage with ChromaDB
- Context-aware retrieval + Gemini-powered answers
- Source page tracking shown alongside each answer
- Chat-style interface, multi-turn Q&A per document

## Architecture
```
PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
                                                            │
                                                            ▼
User Question → Semantic Retrieval → Relevant Chunks → Gemini → Answer
```

## Tech Stack
| Technology | Purpose |
|---|---|
| Gradio | Web UI |
| LangChain | RAG orchestration |
| ChromaDB | Vector database |
| Sentence Transformers | Embedding generation |
| Google Gemini | Answer generation |
| PyPDF | PDF parsing |

## Project Structure
```
.
├── app.py              # Gradio web app — run this to launch
├── requirements.txt    # Dependencies
└── README.md
```

## Environment Variable
This app needs a Google Gemini API key:
```
GOOGLE_API_KEY=your_google_api_key
```
- **Locally:** put this in a `.env` file in the project root.
- **Hugging Face Spaces:** add it under Space → Settings → Variables and
  secrets → New secret (name: `GOOGLE_API_KEY`). Never commit your API key
  to the repo.

## Run Locally
```bash
git clone https://github.com/NavadeepReddy21/PDF-RAG-Assistant.git
cd PDF-RAG-Assistant
pip install -r requirements.txt
# add GOOGLE_API_KEY to a .env file
python app.py
```

## How It Works
1. The uploaded PDF is parsed and split into overlapping text chunks.
2. Each chunk is converted into a vector embedding.
3. Embeddings are stored in a ChromaDB vector store.
4. A user question is embedded and matched against the stored chunks via
   semantic similarity search.
5. The top matching chunks are passed to Gemini as context.
6. Gemini generates an answer, and the source page numbers are shown
   alongside it for transparency.

## Future Enhancements
- Multi-PDF support
- Conversational memory across turns
- Source citation highlighting in the original PDF
- Streaming responses

## License
MIT License

## Author
Navadeep Reddy — [GitHub](https://github.com/NavadeepReddy21)
