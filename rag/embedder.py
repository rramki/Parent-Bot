import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = "data/university_data.txt"
INDEX_PATH = "index/faiss_index.bin"

def build_index():

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        documents = f.read().split("\n\n")

    embeddings = []

    for doc in documents:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        embeddings.append(response.data[0].embedding)

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)

    print("✅ FAISS index built successfully!")

if __name__ == "__main__":
    build_index()
