import os
import tempfile
import streamlit as st

from src import config, document, embeddings, llm

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄")
st.title("📄 PDF RAG Assistant")

if not config.GOOGLE_API_KEY:
    st.warning("Google API Key not found. Please set GOOGLE_API_KEY in your Streamlit secrets.")
    st.stop()

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing PDF..."):
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
                st.success(f"PDF processed successfully! (Loaded {len(documents)} pages, {len(chunks)} chunks)")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                os.unlink(pdf_path)

    if "db" in st.session_state:
        retriever = embeddings.get_retriever(st.session_state.db)
        chat_model = llm.get_llm()

        question = st.text_input("Ask a question about your PDF:")

        if question:
            with st.spinner("Finding answer..."):
                try:
                    # Answer Question
                    answer_stream, source_docs = llm.answer_question_stream(chat_model, retriever, question)
                    
                    st.markdown("### Answer")
                    st.write_stream(answer_stream)
                    
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(source_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Error answering question: {e}")
