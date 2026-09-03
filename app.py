import gradio as gr
from src import config, document, embeddings, llm

# Global state for vector store
db_state = None

def process_file(file_obj):
    global db_state
    if file_obj is None:
        return "Please upload a PDF file."
    
    try:
        documents, chunks = document.process_pdf(file_obj.name)
        db_state = embeddings.create_vector_store(chunks)
        return f"PDF processed successfully! (Loaded {len(documents)} pages, {len(chunks)} chunks)"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def answer_question(question, history):
    global db_state
    if not db_state:
        history.append((question, "Please upload and process a PDF first."))
        return "", history
        
    try:
        retriever = embeddings.get_retriever(db_state)
        chat_model = llm.get_llm()
        
        answer, source_docs = llm.answer_question(chat_model, retriever, question)
        
        sources_text = "\n\n### Source Documents:\n"
        for i, doc in enumerate(source_docs):
            sources_text += f"**Chunk {i+1}:** {doc.page_content}\n\n"
            
        full_response = answer + sources_text
        history.append((question, full_response))
        return "", history
    except Exception as e:
        history.append((question, f"Error: {str(e)}"))
        return "", history

with gr.Blocks(title="PDF RAG Assistant", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📄 PDF RAG Assistant")
    
    if not config.GOOGLE_API_KEY:
        gr.Warning("Google API Key not found. Please set GOOGLE_API_KEY in your Hugging Face Space secrets.")
        
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload a PDF file", file_types=[".pdf"])
            process_btn = gr.Button("Process PDF", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History")
            msg = gr.Textbox(label="Ask a question about your PDF")
            clear = gr.ClearButton([msg, chatbot])
            
    process_btn.click(process_file, inputs=[file_input], outputs=[status_text])
    msg.submit(answer_question, inputs=[msg, chatbot], outputs=[msg, chatbot])

if __name__ == "__main__":
    app.launch()
