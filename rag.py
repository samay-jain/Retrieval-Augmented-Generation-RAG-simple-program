import csv
import os
import json
import requests
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI

# File names for persistence
INDEX_FILE = "space_index.faiss"
DOCS_FILE = "space_docs.json"

# --------------------- Embedding Function ---------------------
# Handles text embedding via Ollama API
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
                print(f"Embedding error for text: '{text[:30]}...': {e}")
                embeddings.append([0.0] * 768)  # Fallback vector in case of error
        return embeddings

# --------------------- Embedding Model Wrapper ---------------------
# Selects and initializes the embedding method based on user input
class EmbeddingModel:
    def __init__(self, model_type="openai"):
        if model_type == "openai":
            self.client = OpenAI(api_key="")
            self.embedding_fn = lambda texts: embedding_functions.OpenAIEmbeddingFunction(api_key="", model_name="text-embedding-3-small")(texts)
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            self.embedding_fn = OllamaEmbeddingFunction()

# --------------------- LLM Wrapper ---------------------
# Selects and initializes the LLM model for chat completions
class LLMModel:
    def __init__(self, model_type="openai"):
        if model_type == "openai":
            self.client = OpenAI(api_key="")
            self.model_name = "gpt-4o-mini"
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
            return f"Error generating response: {str(e)}"

# --------------------- CSV Utilities ---------------------
# Generates a CSV file containing factual data for testing retrieval
# This serves as the source of knowledge for RAG

def generate_csv():
    facts = [
        {"id": 1, "fact": "The Sun makes up more than 99.8% of the total mass of our Solar System."},
        {"id": 2, "fact": "A day on Venus is longer than its year."},
        {"id": 3, "fact": "Neutron stars are so dense that a sugar-cube-sized amount of material would weigh about a billion tons on Earth."},
        {"id": 4, "fact": "There are more stars in the universe than grains of sand on all the Earth's beaches."},
        {"id": 5, "fact": "The footprints left on the Moon by Apollo astronauts will likely remain for millions of years."},
        {"id": 6, "fact": "Mars has the tallest volcano in the solar system — Olympus Mons, which is nearly three times the height of Mount Everest."},
        {"id": 7, "fact": "Jupiter has a magnetic field 14 times stronger than Earth's."},
        {"id": 8, "fact": "Saturn's rings are made mostly of water ice and can stretch up to 282,000 km wide, but are less than a kilometer thick."},
        {"id": 9, "fact": "The Moon is slowly drifting away from Earth at a rate of about 3.8 cm per year."},
        {"id": 10, "fact": "A day on Mercury (one full rotation) lasts about 59 Earth days."},
        {"id": 11, "fact": "In space, astronauts can grow up to 2 inches taller due to the lack of gravity compressing their spines."},
        {"id": 12, "fact": "Pluto, once considered the ninth planet, was reclassified as a dwarf planet in 2006."},
        {"id": 13, "fact": "There is a giant storm on Jupiter, known as the Great Red Spot, which has been raging for at least 400 years."},
        {"id": 14, "fact": "Venus is the hottest planet in the Solar System, even hotter than Mercury, despite being farther from the Sun."},
        {"id": 15, "fact": "One million Earths could fit inside the Sun."},
        {"id": 16, "fact": "Astronauts’ suits cost about $12 million each."},
        {"id": 17, "fact": "The International Space Station travels at about 28,000 kilometers per hour, orbiting Earth roughly every 90 minutes."},
        {"id": 18, "fact": "Black holes warp time and space so much that time slows down near them."},
        {"id": 19, "fact": "A spoonful of a white dwarf star would weigh as much as an elephant."},
        {"id": 20, "fact": "It takes sunlight about 8 minutes and 20 seconds to reach Earth."}
    ]
    with open("space_facts.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "fact"])
        writer.writeheader()
        writer.writerows(facts)
    print("CSV file 'space_facts.csv' created successfully!")

# Loads the CSV data into a list of documents

def load_csv():
    df = pd.read_csv("space_facts.csv", encoding='ISO-8859-1')
    documents = df["fact"].tolist()
    print("\nLoaded documents: ")
    for doc in documents:
        print(f"- {doc}")
    return documents

# --------------------- FAISS Setup ---------------------
# Builds or loads a FAISS index and its associated documents from disk
# If no saved index exists, computes new embeddings and saves them

def setup_faiss(documents, embedding_model):
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        print("\nLoading existing FAISS index and documents...")
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, "r", encoding="utf-8") as f:
            documents = json.load(f)
    else:
        print("\nGenerating new FAISS index...")
        embeddings = embedding_model.embedding_fn(documents)
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        faiss.write_index(index, INDEX_FILE)
        with open(DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f)
    print("\nFAISS index ready!")
    return index, documents

# --------------------- Query Utilities ---------------------
# Retrieves the top-k most similar documents to the query using FAISS

def find_related_chunks(query, faiss_index, documents, embedding_model, top_k=2):
    query_embedding = embedding_model.embedding_fn(query)
    D, I = faiss_index.search(np.array(query_embedding).astype("float32"), top_k)
    print("\nRelated chunks found:")
    return [(documents[i], {"score": D[0][k]}) for k, i in enumerate(I[0])]

# Constructs an augmented prompt for the LLM using retrieved chunks

def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    print("\nAugmented prompt: ")
    print(augmented_prompt)
    return augmented_prompt

# Executes the full Retrieval-Augmented Generation pipeline
# 1. Retrieves relevant docs
# 2. Augments prompt with context
# 3. Generates answer using the LLM

def rag_pipeline(query, faiss_index, documents, llm_model, embedding_model, top_k=2):
    print(f"\nProcessing query: {query}")
    related_chunks = find_related_chunks(query, faiss_index, documents, embedding_model, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)
    response = llm_model.generate_completion([
        {"role": "system", "content": "You are a helpful assistant who only answers based on the context."},
        {"role": "user", "content": augmented_prompt}
    ])
    print("\nGenerated response: ")
    print(response)
    references = [chunk[0] for chunk in related_chunks]
    return response, references

# --------------------- Main ---------------------
# Prompts user to choose LLM and embedding models

def select_models():
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama llama3.2")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
    print("\nSelect Embedding Model:")
    print("1. OpenAI Embeddings")
    print("2. Chroma Default")
    print("3. Nomic Embed Text (Ollama)")
    while True:
        choice = input("Enter choice (1, 2 or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "nomic"}[choice]
            break
    return llm_type, embedding_type

# Main function to coordinate entire RAG demo flow

def main():
    print("Starting the RAG pipeline demo...")
    llm_type, embedding_type = select_models()
    llm_model = LLMModel(llm_type)
    embedding_model = EmbeddingModel(embedding_type)
    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"Using Embeddings: {embedding_type.upper()}")
    generate_csv()
    documents = load_csv()
    faiss_index, documents = setup_faiss(documents, embedding_model)
    queries = [
        "What is relation between a day and a year on Venus?",
        "For how many years will the footprints remain on moon?"
    ]
    for query in queries:
        print("\n" + "=" * 50)
        response, references = rag_pipeline(query, faiss_index, documents, llm_model, embedding_model)
        print("\nFinal Results:")
        print("-" * 30)
        print(f"Response: {response}")
        print("Reference used:")
        for ref in references:
            print(f"- {ref}")
        print("=" * 50)

if __name__ == "__main__":
    main()

