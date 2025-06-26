
import streamlit as st
import csv
import os
import json
import uuid
import requests
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI
import PyPDF2

# ------------------- Constants -------------------
# Define file paths and chunking parameters
INDEX_FILE = "ai_index.faiss"
DOCS_FILE = "ai_docs.json"
CSV_FILE = "ai_facts.csv"
CHUNK_SIZE = 1000            # Maximum number of characters in a chunk
CHUNK_OVERLAP = 200          # Overlap between chunks for better context

# ------------------- Model Selector -------------------
class SimpleModelSelector:
    """Handles selection of LLM and embedding models via Streamlit UI."""

    def __init__(self):
        # Dictionary mapping for supported LLMs
        self.llm_models = {"openai": "GPT-4", "ollama": "llama3.2"}

        # Supported embedding models with their metadata
        self.embedding_models = {
            "openai": {"name": "OpenAI Embeddings", "dimensions": 1536, "model_name": "text-embedding-3-small"},
            "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
            "nomic": {"name": "Nomic Embed Text", "dimensions": 768, "model_name": "nomic-embed-text"},
        }

    def select_models(self):
        """Renders Streamlit sidebar for model selection."""
        st.sidebar.title("Model Selection")
        llm = st.sidebar.radio("Choose LLM Model:", options=list(self.llm_models.keys()), format_func=lambda x: self.llm_models[x])
        embedding = st.sidebar.radio("Choose Embedding Model:", options=list(self.embedding_models.keys()), format_func=lambda x: self.embedding_models[x]["name"])
        return llm, embedding

# ------------------- PDF Processor -------------------
class SimplePDFProcessor:
    """Processes and splits PDF or plain text content into chunks."""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """Extracts text from all pages of a PDF."""
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in reader.pages)

    def create_chunks(self, text, pdf_file):
        """Divides large text into overlapping chunks with optional trimming to sentence end."""
        chunks, start = [], 0
        while start < len(text):
            end = start + self.chunk_size
            if start > 0:
                start -= self.chunk_overlap
            chunk = text[start:end]
            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            chunks.append({"id": str(uuid.uuid4()), "text": chunk, "metadata": {"source": pdf_file.name}})
            start = end
        return chunks

# ------------------- Ollama Embedding Wrapper -------------------
class OllamaEmbeddingFunction:
    """Handles embedding generation via Ollama local API."""

    def __init__(self, base_url="http://localhost:11434/api/embeddings", model="nomic-embed-text"):
        self.base_url = base_url
        self.model = model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            payload = {"model": self.model, "prompt": text}
            try:
                response = requests.post(self.base_url, json=payload)
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
            except Exception as e:
                print(f"Embedding error for text: '{text[:30]}...': {e}")
                embeddings.append([0.0] * 768)
        return embeddings

# ------------------- Core RAG System -------------------
class SimpleRAGSystem:
    """Handles vector indexing, chunk retrieval, and LLM-powered response generation."""

    def __init__(self, embedding_model="openai", llm_model="openai"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.setup_embedding_function()
        self.llm = OpenAI() if llm_model == "openai" else OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.collection = self.setup_collection()

    def setup_embedding_function(self):
        """Initializes the embedding function depending on the selected model."""
        try:
            if self.embedding_model == "openai":
                self.embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")
            elif self.embedding_model == "nomic":
                self.embedding_fn = OllamaEmbeddingFunction()
            else:
                raise ValueError("Unsupported embedding model")
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e

    def setup_collection(self):
        """Loads previously saved FAISS index and documents."""
        if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
            index = faiss.read_index(INDEX_FILE)
            with open(DOCS_FILE, "r", encoding="utf-8") as f:
                documents = json.load(f)
            return index, documents
        return None

    def setup_faiss(self, documents):
        """Creates and saves a FAISS index from the given documents."""
        embeddings = self.embedding_fn([doc["text"] for doc in documents])
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(index, INDEX_FILE)
        with open(DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f)
        self.collection = (index, documents)
        return index, documents

    def find_related_chunks(self, query, faiss_index, documents, top_k=2):
        """Finds top-k most similar document chunks to the query using FAISS."""
        query_embedding = self.embedding_fn(query)
        D, I = faiss_index.search(np.array(query_embedding).astype("float32"), top_k)
        return [(documents[i], {"score": D[0][k]}) for k, i in enumerate(I[0])]

    def generate_response(self, query, context):
        """Sends prompt and context to LLM and returns generated answer."""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say no, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini" if self.llm_model == "openai" else "llama3.2",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self):
        """Returns information about the selected embedding model."""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }

# ------------------- File Loader with Caching -------------------
@st.cache_data(show_spinner=False)
def load_file_text(file):
    """Extracts text content from supported file formats."""
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        from docx import Document
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

# ------------------- Main Streamlit UI -------------------
def main():
    """Main function for Streamlit UI and user interaction."""
    st.title("Simple RAG System")

    # Initialize Streamlit session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    # Sidebar: Model selection
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    # Reset state if embedding model changes
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None
        st.warning("Embedding model changed. Please re-upload your documents.")

    # Initialize or retrieve RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(f"Current Embedding Model:\n- Name: {embedding_info['name']}\n- Dimensions: {embedding_info['dimensions']}")
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # Upload interface for PDF, TXT, DOCX
    pdf_file = st.file_uploader("Upload File", type=["pdf", "txt", "docx"])

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        processor = SimplePDFProcessor()
        with st.spinner("Processing document..."):
            try:
                text = load_file_text(pdf_file)
                chunks = processor.create_chunks(text, pdf_file)
                if st.session_state.rag_system.setup_faiss(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"Successfully processed: {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Query interface
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("Query your documents")
        query = st.text_input("Ask a question: ")

        if query:
            with st.spinner("Generating response..."):
                results = st.session_state.rag_system.find_related_chunks(
                    query,
                    st.session_state.rag_system.collection[0],
                    st.session_state.rag_system.collection[1]
                )
                if results:
                    context = "\n\n".join([r[0]["text"] for r in results])
                    response = st.session_state.rag_system.generate_response(query, context)
                    if response:
                        st.markdown("### Answer:")
                        st.write(response)
                        with st.expander("View Source Passages"):
                            for idx, (doc, _) in enumerate(results, 1):
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc["text"])
        else:
            st.info("Please upload a document to get started!")

# Entry point
if __name__ == "__main__":
    main()
