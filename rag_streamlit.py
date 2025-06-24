import streamlit as st
import csv
import os
import json
import requests
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI

# ------------------- Constants -------------------
INDEX_FILE = "ai_index.faiss"
DOCS_FILE = "ai_docs.json"
CSV_FILE = "ai_facts.csv"

# ------------------- Embedding Function for Ollama -------------------
class OllamaEmbeddingFunction:
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
                print(f"Embedding error: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

# ------------------- Embedding Model Wrapper -------------------
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        if model_type == "openai":
            self.client = OpenAI(api_key="")
            self.embedding_fn = lambda texts: [self.client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding for text in texts]
        elif model_type == "chroma":
            self.embedding_fn = lambda texts: [[0.0] * 768 for _ in texts]
        elif model_type == "nomic":
            self.embedding_fn = OllamaEmbeddingFunction()

# ------------------- LLM Model Wrapper -------------------
class LLMModel:
    def __init__(self, model_type="openai"):
        if model_type == "openai":
            self.client = OpenAI(api_key="")
            self.model_name = "gpt-4o"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

# ------------------- CSV Utilities -------------------
def generate_csv():
    if os.path.exists(CSV_FILE):
        return
    facts = [
        {"id": i + 1, "fact": fact} for i, fact in enumerate([
            "Artificial Intelligence is a branch of computer science that aims to create intelligent machines.",
            "Alan Turing is considered the father of Artificial Intelligence.",
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Deep learning is inspired by the structure and function of the human brain.",
            "The Turing Test evaluates a machine's ability to exhibit human-like intelligence.",
            "AI is widely used in speech recognition systems like Siri and Alexa.",
            "Natural Language Processing (NLP) helps computers understand human language.",
            "IBM Watson defeated two Jeopardy champions in 2011.",
            "Self-driving cars rely heavily on AI algorithms to operate safely.",
            "AlphaGo, developed by DeepMind, defeated a world champion Go player.",
            "AI can analyze massive datasets faster than humans.",
            "Chatbots are a common application of conversational AI.",
            "Reinforcement learning teaches agents to make decisions by trial and error.",
            "Computer vision enables machines to interpret and understand images.",
            "Facial recognition is a form of AI used in security and social media.",
            "Generative AI models can create new text, images, music, and more.",
            "GPT-3 by OpenAI is a powerful language generation model.",
            "AI is used in healthcare for diagnostics, treatment recommendations, and drug discovery.",
            "Bias in AI models can lead to unfair outcomes.",
            "Training large AI models requires significant computational power.",
            "Ethical AI development ensures transparency, fairness, and accountability.",
            "AI in finance is used for fraud detection and algorithmic trading.",
            "Autonomous drones use AI for navigation and decision-making.",
            "AI models can be trained using supervised, unsupervised, or reinforcement learning.",
            "Explainable AI aims to make AI decisions understandable by humans.",
            "AI-powered recommendation systems are used by Netflix, Amazon, and YouTube.",
            "AI is being used to fight climate change by optimizing energy use.",
            "OpenAI’s mission is to ensure AGI benefits all of humanity.",
            "Robotics often integrates AI for movement, perception, and planning.",
            "AI can enhance accessibility, such as automatic captioning for the hearing impaired."
        ])
    ]
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)

def load_csv():
    df = pd.read_csv(CSV_FILE, encoding="ISO-8859-1")
    return df["fact"].tolist()

# ------------------- FAISS Setup -------------------
def setup_faiss(documents, embedding_model):
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        embeddings = embedding_model.embedding_fn(documents)
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(index, INDEX_FILE)
        with open(DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f)
    return index, documents

# ------------------- RAG Pipeline -------------------
def find_related_chunks(query, faiss_index, documents, embedding_model, top_k=2):
    query_embedding = embedding_model.embedding_fn([query])
    D, I = faiss_index.search(np.array(query_embedding).astype("float32"), top_k)
    return [(documents[i], {"score": D[0][k]}) for k, i in enumerate(I[0])]

def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

def rag_pipeline(query, faiss_index, documents, llm_model, embedding_model, top_k=2):
    related_chunks = find_related_chunks(query, faiss_index, documents, embedding_model, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)
    response = llm_model.generate_completion([
        {"role": "system", "content": "You are a helpful assistant who only answers based on the context."},
        {"role": "user", "content": augmented_prompt}
    ])
    references = [chunk[0] for chunk in related_chunks]
    return response, references, augmented_prompt

# ------------------- Streamlit UI -------------------
def streamlit_app():
    st.set_page_config(page_title="AI Facts RAG", layout="wide")
    st.title("🧠 AI Facts RAG System")

    llm_type = st.sidebar.radio(
        "Select LLM Model:", ["openai", "ollama"],
        format_func=lambda x: "OpenAI GPT-4" if x == "openai" else "Ollama llama3.2"
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:", ["openai", "chroma", "nomic"],
        format_func=lambda x: {
            "openai": "OpenAI Embeddings",
            "chroma": "Chroma Default",
            "nomic": "Nomic Embed Text (Ollama)"
        }[x]
    )

    # Load facts
    generate_csv()
    documents = load_csv()

    # Setup or re-setup
    if ("initialized" not in st.session_state or
        st.session_state.last_llm_type != llm_type or
        st.session_state.last_embedding_type != embedding_type):

        embedding_model = EmbeddingModel(embedding_type)
        llm_model = LLMModel(llm_type)
        index, documents = setup_faiss(documents, embedding_model)

        st.session_state.embedding_model = embedding_model
        st.session_state.llm_model = llm_model
        st.session_state.index = index
        st.session_state.documents = documents
        st.session_state.last_llm_type = llm_type
        st.session_state.last_embedding_type = embedding_type
        st.session_state.initialized = True

    # 🔽 Dropdown to show all facts
    st.markdown("### 📋 Explore Available AI Facts")
    selected_fact = st.selectbox("Select a fact to view:", documents)
    st.info(selected_fact)

    # 🔎 Input box for custom queries
    query = st.text_input("Enter your question about AI:", placeholder="e.g., What is deep learning?")

    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = rag_pipeline(
                query,
                st.session_state.index,
                st.session_state.documents,
                st.session_state.llm_model,
                st.session_state.embedding_model
            )

        # Display result
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🤖 Response")
            st.write(response)
        with col2:
            st.markdown("### 📚 References Used")
            for ref in references:
                st.markdown(f"- {ref}")

        with st.expander("Technical Details"):
            st.markdown("#### Augmented Prompt")
            st.code(augmented_prompt +" "+ response)
            st.markdown("#### Model Configuration")
            st.markdown(f"- LLM Model: `{llm_type.upper()}`")
            st.markdown(f"- Embedding Model: `{embedding_type.upper()}`")

# Run app
if __name__ == "__main__":
    streamlit_app()
