"""
app.py — PDF RAG Assistant
Manual RAG pipeline (no langchain.chains) to avoid version conflicts.
- PyPDFLoader + RecursiveCharacterTextSplitter  (langchain-community / text-splitters)
- GoogleGenerativeAIEmbeddings                  (langchain-google-genai)
- ChromaDB                                      (chromadb)
- Gemini 2.5 Flash via ChatGoogleGenerativeAI  (langchain-google-genai)
"""

import subprocess, sys

# Runtime install — bypasses HF Docker cache entirely
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "langchain-community",
    "langchain-text-splitters",
    "langchain-google-genai",
    "langchain-huggingface",
    "chromadb",
    "pypdf",
    "python-dotenv",
    "sentence-transformers",
])

import os
import tempfile
import types

# audioop shim — removed in Python 3.13 but pydub/gradio still import it
if "audioop" not in sys.modules:
    try:
        import audioop  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["audioop"] = types.ModuleType("audioop")

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

load_dotenv()
print(f"[debug] Gradio version: {gr.__version__}")

# ===================== HELPERS =====================
def get_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    print(f"[debug] GOOGLE_API_KEY present: {bool(key)}, len: {len(key) if key else 0}")
    return key

state = {"vectorstore": None, "filename": None, "embeddings": None}

def get_embeddings(api_key=None):
    if state["embeddings"] is None:
        print("Loading local embedding model (all-MiniLM-L6-v2)...")
        state["embeddings"] = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embedding model ready.")
    return state["embeddings"]

# ===================== CORE FUNCTIONS =====================
def build_index(pdf_file):
    if pdf_file is None:
        return "⚠️ Please upload a PDF first.", gr.update(interactive=False)

    api_key = get_api_key()
    if not api_key:
        return (
            "❌ GOOGLE_API_KEY not set. Go to Space → Settings → "
            "Variables and secrets → add GOOGLE_API_KEY → restart.",
            gr.update(interactive=False),
        )

    try:
        pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
        pages = PyPDFLoader(pdf_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        ).split_documents(pages)

        persist_dir = tempfile.mkdtemp(prefix="chroma_")
        state["vectorstore"] = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory=persist_dir,
        )
        state["filename"] = os.path.basename(pdf_path)

        return (
            f"✅ **{state['filename']}** ready — "
            f"{len(pages)} pages, {len(chunks)} chunks. Ask away!",
            gr.update(interactive=True),
        )
    except Exception as e:
        return f"❌ Error: {e}", gr.update(interactive=False)


SMALL_TALK = {"hi", "hello", "hey", "hii", "helo", "sup", "yo", "good morning",
              "good afternoon", "good evening", "how are you", "what's up", "whats up"}

def is_small_talk(text: str) -> bool:
    return text.strip().lower().rstrip("!?.") in SMALL_TALK


def answer_question(question, history):
    if state["vectorstore"] is None:
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "⚠️ Please upload and process a PDF first."}
        ], ""
    if not question.strip():
        return history, ""

    try:
        api_key = get_api_key()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

        if is_small_talk(question):
            prompt = (
                f"You are a helpful AI assistant. The user said: '{question}'. "
                "Respond in a friendly, brief way. If a document is uploaded, "
                "let them know they can ask questions about it too."
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        else:
            docs = state["vectorstore"].similarity_search(question, k=3)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = (
                "You are a helpful AI assistant with two sources of knowledge:\n"
                "1. The document context below\n"
                "2. Your own general knowledge\n\n"
                "If the question can be answered from the document context, "
                "answer using ONLY the context and mention it came from the document.\n"
                "If the context is not relevant to the question, answer from your "
                "general knowledge normally — do NOT say the answer isn't in the context.\n\n"
                f"Document context:\n{context}\n\n"
                f"Question: {question}"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

            # Only show source pages if answer came from the document
            pages_used = sorted({d.metadata.get("page", "?") for d in docs})
            pages_display = ", ".join(
                str(p + 1 if isinstance(p, int) else p) for p in pages_used
            )
            if any(kw in answer.lower() for kw in ["document", "according", "context", "resume", "pdf"]):
                answer += f"\n\n*Source page(s): {pages_display}*"

    except Exception as e:
        answer = f"❌ Error: {e}"

    return history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ], ""


def clear_session():
    state["vectorstore"] = None
    state["filename"] = None
    state["embeddings"] = None
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
    border-radius: 16px; padding: 28px 32px; margin-bottom: 20px;
    box-shadow: 0 4px 16px rgba(67,56,202,0.25);
}
#app-header h1 { color:white !important; font-size:26px !important; font-weight:700 !important; margin-bottom:6px !important; }
#app-header p  { color:rgba(255,255,255,0.9) !important; font-size:14px !important; margin:0 !important; }
.panel-title   { font-size:14px !important; font-weight:600 !important; color:#4338CA !important;
                 text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px !important; }
footer { visibility: hidden }
"""

# ===================== UI =====================
with gr.Blocks(title="PDF RAG Assistant") as demo:

    with gr.Column(elem_id="app-header"):
        gr.HTML(
            "<h1>📄 PDF RAG Assistant</h1>"
            "<p>Upload a PDF and ask questions — powered by semantic search and Google Gemini</p>"
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("Document", elem_classes="panel-title")
            pdf_in    = gr.File(label="Upload PDF", file_types=[".pdf"])
            index_btn = gr.Button("📑 Process PDF", variant="primary")
            status    = gr.Markdown("Upload a PDF to get started.")
            clear_btn = gr.Button("🗑️ Clear Session")

        with gr.Column(scale=2, min_width=420):
            gr.Markdown("Chat", elem_classes="panel-title")
            chatbot = gr.Chatbot(height=420, label=None)
            with gr.Row():
                question_in = gr.Textbox(
                    placeholder="Ask something about the document...",
                    show_label=False, scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1, interactive=False)

    index_btn.click(build_index, inputs=[pdf_in], outputs=[status, ask_btn])
    ask_btn.click(answer_question, inputs=[question_in, chatbot], outputs=[chatbot, question_in])
    question_in.submit(answer_question, inputs=[question_in, chatbot], outputs=[chatbot, question_in])
    clear_btn.click(clear_session, outputs=[chatbot, status])

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)
else:
    demo.launch(theme=theme, css=custom_css)
