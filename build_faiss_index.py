import os
import faiss
from sentence_transformers import SentenceTransformer

# Load bge-m3 embedding model
embedder = SentenceTransformer('BAAI/bge-m3')

folder_path = "news_articles"
docs = []
filenames = []

for fname in os.listdir(folder_path):
    if fname.endswith(".txt"):
        with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
            content = f.read()
            docs.append("passage: " + content)  # Important: add "passage: "
            filenames.append(fname)

# Encode documents
embeddings = embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

# Create FAISS index
dimension = embeddings.shape[1]  # should be 1024 for bge-m3
index = faiss.IndexFlatIP(dimension)  # Use inner product (IP) with normalized embeddings
index.add(embeddings)

# Save index and filenames
faiss.write_index(index, "news_index.faiss")
with open("news_filenames.txt", "w", encoding="utf-8") as f:
    for name in filenames:
        f.write(name + "\n")

print("✅ Rebuilt FAISS index with bge-m3.")