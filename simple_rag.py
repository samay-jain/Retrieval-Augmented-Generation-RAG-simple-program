import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from openai import OpenAI


import requests

class OllamaEmbeddingFunction:
    def __init__(self, base_url="http://localhost:11434/api/embeddings", model="nomic-embed-text"):
        self.base_url = base_url
        self.model = model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]

        embeddings = []
        for text in input:
            payload = {
                "model": self.model,
                "prompt": text,
            }
            try:
                response = requests.post(f"{self.base_url}", json=payload)
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
            except Exception as e:
                print(f"Embedding error for text: '{text[:30]}...': {e}")
                embeddings.append([0.0] * 768)  # or whatever your fallback size is
        return embeddings



# Using 'nomic-embed-text' model for Ollama
# API key for OpenAI - paid
api_key = ""

class EmbeddingModel:
    def __init__(self, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key)
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key,
                model_name = "text-embedding-3-small"
            )
        elif model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            # using Ollama nomic-embed-text model
            # self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            #     api_key="ollama",
            #     api_base="http://localhost:11434/v1",
            #     model_name="nomic-embed-text"
            # )
            self.embedding_fn = OllamaEmbeddingFunction()

class LLMModel:
    def __init__(self, model_type = "openai"):
        self.model_type = model_type
        if model_type == "openai":
            self.client = OpenAI(api_key="")
            self.model_name = "gpt-4o-mini"
        else:
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.model_name = "llama3.2"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                temperature = 0.0, # 0.0 is deterministic
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
def select_models():
    # Select LLM Model
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama llama2")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
        print("Please enter either 1 or 2")

    # Select Embedding Model
    print("\nSelect Embedding Model: ")
    print("1. OpenAI Embeddings")
    print("2. Chroma Default")
    print("3. Nomic Embed Text (Ollama)")

    while True:
        choice  = input("Enter choice (1, 2 or 3): ").strip()
        if(choice in ["1", "2", "3"]):
            embedding_type = {"1": "openai", "2": "chroma", "3":"nomic"}[choice]
            break
        print("Please enter 1, 2 or 3")

    return llm_type, embedding_type


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

def load_csv():
    df = pd.read_csv("space_facts.csv", encoding='ISO-8859-1')

    documents = df["fact"].tolist()
    print("\nLoaded documents: ")
    for doc in documents:
        print(f"- {doc}")
    return documents

# def setup_chromadb(documents, embedding_model):
#     from chromadb import PersistentClient

#     client = PersistentClient(path="chroma_store")
#     # client = chromadb.Client()
#     try:
#         client.delete_collection("space_facts")
#     except Exception as e:
#         print("Exception occured: ", e)
        

#     collection = client.create_collection(
#         name="space_facts",
#         embedding_function= embedding_model.embedding_fn
#     )

#     collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

#     print("\nDocuments added to ChromaDB collection successfully!")
#     return collection


def setup_chromadb(documents, embedding_model):
    from chromadb import PersistentClient

    client = PersistentClient(path="chroma_store")

    # Get all existing collections and check if "space_facts" exists
    existing_collections = [col.name for col in client.list_collections()]
    if "space_facts" in existing_collections:
        print("Collection already exists. Deleting it first.")
        client.delete_collection("space_facts")

    # Now re-create it
    collection = client.create_collection(
        name="space_facts",
        embedding_function=embedding_model.embedding_fn
    )

    collection.add(
        documents=documents,
        ids=[str(i) for i in range(len(documents))],
        metadatas=[{"index": i} for i in range(len(documents))]
    )

    print("\nDocuments added to ChromaDB collection successfully!")
    return collection


def find_related_chunks(query, collection, top_k=2):
    results = collection.query(query_texts=[query], n_results=top_k)

    print("\nRelated chunks found:")
    for doc in results["documents"][0]:
        print(f"- {doc}")

    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]
                if results["metadatas"][0]
                else [{}] * len(results["documents"][0])
            ),
        )
    )

def augment_prompt(query, related_chunks):
    context = "\n".join([chunk[0] for chunk in related_chunks])
    augmented_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    print("\nAugmented prompt: ")
    print(augmented_prompt)

    return augmented_prompt

def rag_pipeline(query, collection, llm_model, top_k=2):
    print(f"\nProcessing query: {query}")

    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer questions about space but only answers questions that are directly related to the sources/documents given."
            },
            {
                "role": "user",
                "content": augmented_prompt
            }
        ]
    )

    print("\nGenerated response: ")
    print(response)

    references = [chunk[0] for chunk in related_chunks]
    return response, references

def main():
    print("Starting the RAG pipeline demo...")

    # Select models
    llm_type, embedding_type = select_models()

    # Initialize models
    llm_model = LLMModel(llm_type)
    embedding_model = EmbeddingModel(embedding_type)

    print(f"\nUsing LLM: {llm_type.upper()}")
    print(f"Using Embeddings: {embedding_type.upper()}")

    # Generate and load data
    generate_csv()
    documents = load_csv()

    # Setup ChromaDB
    collection = setup_chromadb(documents, embedding_model)

    # Run queries
    queries = [
        "What is relation between a day and a year on Venus?",
        "For how many years will the footprints remain on moon?"
    ]

    for query in queries:
        print("\n" + "=" * 50)
        print(f"Processing query: {query}")
        response, references = rag_pipeline(query, collection, llm_model)

        print("\nFinal Results: ")
        print("-" * 30)
        print(f"Response: {response}")
        print(f"Reference used:")

        for ref in references:
            print(f"- {ref}")
        print("=" * 50)

if __name__ == "__main__":
    main()