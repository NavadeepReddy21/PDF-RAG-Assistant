import os
import tempfile
import streamlit as st

from src import config, document, embeddings, llm

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")

# Inject Custom Enterprise CSS
st.markdown("""
    <style>
        /* Hide Streamlit default header and footer */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Modern aesthetic adjustments */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 900px;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f4fcfb !important;
            border-right: 1px solid #e0f2f1;
        }
        
        /* Chat Input Styling */
        .stChatInputContainer {
            padding-bottom: 2rem;
            background-color: transparent !important;
        }
        
        /* Custom Chat Bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        
        /* Assistant Bubble (Light Grey) */
        [data-testid="stChatMessage"]:has([data-testid="stIconMaterial"][title="assistant"]) {
            background-color: #f1f3f4;
            color: #333333;
            border: none;
            border-bottom-left-radius: 4px;
            margin-right: 20%;
        }

        /* User Bubble (Teal) */
        [data-testid="stChatMessage"]:has([data-testid="stIconMaterial"][title="user"]) {
            background-color: #26a69a;
            color: white !important;
            border: none;
            border-bottom-right-radius: 4px;
            margin-left: 20%;
        }
        
        /* Ensure user text is white */
        [data-testid="stChatMessage"]:has([data-testid="stIconMaterial"][title="user"]) p {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📄 PDF RAG Assistant")

if not config.GOOGLE_API_KEY:
    st.warning("Google API Key not found. Please set GOOGLE_API_KEY in your Streamlit secrets.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Configuration & File Upload
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing Document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name

                try:
                    # Process Document
                    documents, chunks = document.process_pdf(pdf_path)
                    
                    # Create Vector Store
                    db = embeddings.create_vector_store(chunks)
                    
                    st.session_state.db = db
                    st.session_state.current_file = uploaded_file.name
                    
                    # Optional: Add a system message saying the PDF is ready
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ Successfully loaded **{uploaded_file.name}** ({len(documents)} pages, {len(chunks)} chunks). What would you like to know about this document?"
                    })
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.unlink(pdf_path)
    else:
        st.info("Please upload a PDF document to begin.")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Source Documents"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)

# Handle new user input
if prompt := st.chat_input("Ask a question about your PDF..."):
    if "db" not in st.session_state:
        st.warning("Please upload a PDF in the sidebar first.")
    else:
        # 1. Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever = embeddings.get_retriever(st.session_state.db)
                    chat_model = llm.get_llm()
                    
                    answer_stream, source_docs = llm.answer_question_stream(chat_model, retriever, prompt)
                    
                    # Stream the response
                    full_response = st.write_stream(answer_stream)
                    
                    # Add expander for sources in the UI directly
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": source_docs
                    })
                    
                except Exception as e:
                    st.error(f"Error answering question: {e}")
