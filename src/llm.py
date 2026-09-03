from langchain_google_genai import ChatGoogleGenerativeAI
from . import config

def get_llm():
    """Initializes and returns the Chat Gemini model."""
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL, 
        google_api_key=config.GOOGLE_API_KEY
    )

def answer_question_stream(llm, retriever, question: str):
    """Retrieves context and yields the LLM answer stream."""
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful assistant. Answer only using the context below.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    
    def generate():
        for chunk in llm.stream(prompt):
            yield chunk.content
            
    return generate(), docs
