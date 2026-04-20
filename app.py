import os
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return self.text_splitter.split_documents(docs)

class VectorStoreManager:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None

    def create_vector_store(self, document_chunks):
        self.vector_store = FAISS.from_documents(document_chunks, self.embeddings)
        return self.vector_store

    def get_retriever(self, k=3):
        if not self.vector_store:
            raise ValueError("Vector store has not been created yet.")
        return self.vector_store.as_retriever(search_kwargs={"k": k})

class RAGPipeline:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        # Fetch the key securely from Hugging Face Secrets
        MY_GROQ_KEY = os.environ.get("GROQ_API_KEY")
        # -------------------------------
        
        self.llm = ChatGroq(
            temperature=0.2, 
            model_name=model_name,
            groq_api_key=MY_GROQ_KEY  # We are forcing the key directly into the LLM
        )
        self.qa_chain = None
        self._setup_prompt()

    def _setup_prompt(self):
        system_prompt = (
            "You are a helpful and knowledgeable AI assistant. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If the answer is not contained within the context, say 'I don't have enough information to answer that based on the provided document.' "
            "Do not make up information. Keep your answers clear and concise.\n\n"
            "Context: {context}"
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

    def initialize_chain(self, retriever):
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    def answer_question(self, question):
        if not self.qa_chain:
            return "Please upload and process a document first."
        response = self.qa_chain.invoke({"input": question})
        return response["answer"]

class ChatbotApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vs_manager = VectorStoreManager()
        self.rag_pipeline = RAGPipeline()
    
    def process_file(self, file):
        if file is None:
            return "Please upload a PDF file."
        try:
            chunks = self.doc_processor.process_pdf(file.name)
            self.vs_manager.create_vector_store(chunks)
            retriever = self.vs_manager.get_retriever()
            self.rag_pipeline.initialize_chain(retriever)
            return f"Success! Document processed into {len(chunks)} chunks. You can now ask questions."
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def chat(self, message, history):
        # We added a try-except block here to catch the exact error!
        try:
            return self.rag_pipeline.answer_question(message)
        except Exception as e:
            return f"❌ Backend Error: {str(e)}"

    def launch(self):
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🚀 Groq + LLaMA 3 RAG Chatbot")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                    upload_button = gr.Button("Process Document")
                    status_text = gr.Textbox(label="System Status", interactive=False)
                
                with gr.Column(scale=2):
                    chatbot = gr.ChatInterface(
                        fn=self.chat,
                        title="Document Q&A",
                        fill_height=True
                    )
            
            upload_button.click(
                fn=self.process_file,
                inputs=[file_input],
                outputs=[status_text]
            )

        demo.launch(debug=True)

if __name__ == "__main__":
    app = ChatbotApp()
    app.launch()
