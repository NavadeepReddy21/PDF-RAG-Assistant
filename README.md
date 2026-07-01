---
title: PDF RAG Assistant
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 📄 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload a PDF
and ask questions about it in natural language. Combines semantic search
with Google's Gemini model for accurate, context-aware answers.

🔗 **Live demo:** [https://navadeep-21-pdf-rag-assistant.hf.space/]

## Features
- Upload any PDF directly in the browser
- Intelligent text chunking with overlap for better context retention
- Semantic embeddings via Google Generative AI (`embedding-001`)
- Vector storage with ChromaDB
- Context-aware retrieval + Gemini-powered answers (`gemini-2.5-flash`)
- Source page tracking shown alongside each answer
- Chat-style interface, multi-turn Q&A per document
- Supports both standard (AIza...) and new Auth (AQ...) Gemini API keys

## Architecture
```
PDF Upload → Text Extraction → Chunking → Google Embeddings → ChromaDB
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
| Google Generative AI Embeddings | Embedding generation |
| Google Gemini 2.5 Flash | Answer generation |
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
- **Hugging Face Spaces:** Space → Settings → Variables and secrets →
  New secret → Name: `GOOGLE_API_KEY` → paste your key → Save, then restart.
  Never commit your API key to the repo.

## Run Locally
```bash
git clone https://github.com/NavadeepReddy21/PDF-RAG-Assistant.git
cd PDF-RAG-Assistant
pip install -r requirements.txt
# add GOOGLE_API_KEY=your_key to a .env file
python app.py
```

## How It Works
1. The uploaded PDF is parsed page-by-page with PyPDFLoader.
2. Text is split into overlapping chunks (500 chars, 50 overlap).
3. Each chunk is converted into a vector embedding via Google's embedding model.
4. Embeddings are stored in a ChromaDB vector store.
5. A user question is embedded and matched against stored chunks via cosine similarity.
6. The top-3 matching chunks are passed to Gemini as context.
7. Gemini generates an answer grounded in the retrieved content.
8. Source page numbers are shown alongside each answer for transparency.

## Future Enhancements
- Multi-PDF support
- Conversational memory across turns
- Source citation highlighting in the original PDF
- Streaming responses

## License
MIT License

## Author
Navadeep Reddy — [GitHub](https://github.com/NavadeepReddy21)
