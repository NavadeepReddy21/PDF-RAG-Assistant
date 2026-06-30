"""
app.py
PDF RAG Assistant - Gradio Web App

Upload a PDF, ask questions about it in natural language. Uses:
- PyPDFLoader for parsing
- Sentence Transformers (all-MiniLM-L6-v2) for embeddings
- ChromaDB for vector storage
- Google Gemini for answer generation

Requires a GOOGLE_API_KEY (set as a secret in your deployment environment,
e.g. Hugging Face Space Settings -> Variables and secrets).
"""

import os
import shutil
import tempfile

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Loaded once at startup - embeddings model is the slow part to initialize
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model ready.")

# Per-session state holds the QA chain built for the currently uploaded PDF
state = {"qa_chain": None, "filename": None}


def build_index(pdf_file):
    """Process an uploaded PDF into a retriever-backed QA chain."""
    if pdf_file is None:
        return "⚠️ Please upload a PDF first.", gr.update(interactive=False)

    if not GOOGLE_API_KEY:
        return (
            "❌ GOOGLE_API_KEY is not set. Add it as a secret in your "
            "deployment environment (e.g. Hugging Face Space → Settings → "
            "Variables and secrets) and restart the Space.",
            gr.update(interactive=False),
        )

    try:
        loader = PyPDFLoader(pdf_file.name)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        # Use a fresh temp directory per upload so old indexes don't bleed in
        persist_dir = tempfile.mkdtemp(prefix="chroma_")
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=persist_dir
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True
        )

        state["qa_chain"] = qa_chain
        state["filename"] = os.path.basename(pdf_file.name)

        return (
            f"✅ **{state['filename']}** indexed — {len(pages)} pages, "
            f"{len(chunks)} chunks. Ask a question below.",
            gr.update(interactive=True),
        )

    except Exception as e:
        return f"❌ Error processing PDF: {e}", gr.update(interactive=False)


def answer_question(question, history):
    if state["qa_chain"] is None:
        history = history + [(question, "Please upload and index a PDF first.")]
        return history, ""

    if not question.strip():
        return history, ""

    result = state["qa_chain"].invoke({"query": question})
    answer = result["result"]

    sources = result.get("source_documents", [])
    pages_used = sorted({doc.metadata.get("page", "?") for doc in sources})
    if pages_used:
        pages_display = ", ".join(str(p + 1 if isinstance(p, int) else p) for p in pages_used)
        answer += f"\n\n*Source page(s): {pages_display}*"

    history = history + [(question, answer)]
    return history, ""


def clear_session():
    state["qa_chain"] = None
    state["filename"] = None
    return [], "Upload a PDF to get started."


# ===================== STYLING =====================
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_shadow="0 1px 3px rgba(0,0,0,0.08)",
    block_radius="16px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
)

custom_css = """
#app-header {
    background: linear-gradient(135deg, #4338CA 0%, #7C3AED 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(67, 56, 202, 0.25);
}
#app-header h1 { color: white !important; font-size: 26px !important; font-weight: 700 !important; margin-bottom: 6px !important; }
#app-header p { color: rgba(255,255,255,0.9) !important; font-size: 14px !important; margin: 0 !important; }
.panel-title { font-size: 14px !important; font-weight: 600 !important; color: #4338CA !important; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px !important; }
footer {visibility: hidden}
"""

# ===================== UI =====================
with gr.Blocks(title="PDF RAG Assistant", theme=theme, css=custom_css) as demo:

    with gr.Column(elem_id="app-header"):
        gr.HTML(
            "<h1>📄 PDF RAG Assistant</h1>"
            "<p>Upload a PDF and ask questions about it — powered by "
            "semantic search and Google Gemini</p>"
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("Document", elem_classes="panel-title")
            pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"])
            index_btn = gr.Button("📑 Process PDF", variant="primary")
            status = gr.Markdown("Upload a PDF to get started.")
            clear_btn = gr.Button("🗑️ Clear Session")

        with gr.Column(scale=2, min_width=420):
            gr.Markdown("Chat", elem_classes="panel-title")
            chatbot = gr.Chatbot(height=420, label=None)
            with gr.Row():
                question_in = gr.Textbox(
                    placeholder="Ask something about the document...",
                    show_label=False,
                    scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1, interactive=False)

    index_btn.click(build_index, inputs=[pdf_in], outputs=[status, ask_btn])
    ask_btn.click(answer_question, inputs=[question_in, chatbot], outputs=[chatbot, question_in])
    question_in.submit(answer_question, inputs=[question_in, chatbot], outputs=[chatbot, question_in])
    clear_btn.click(clear_session, outputs=[chatbot, status])

if __name__ == "__main__":
    demo.launch()
