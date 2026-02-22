import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = "data/university_data.txt"
INDEX_PATH = "index/faiss_index.bin"

# Load documents
with open(DATA_PATH, "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

def retrieve_context(question, k=3):

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    D, I = index.search(np.array([query_embedding]), k)

    retrieved_docs = [documents[i] for i in I[0]]
    return "\n".join(retrieved_docs)
