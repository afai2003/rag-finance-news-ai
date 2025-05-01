from fastapi import FastAPI
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

app = FastAPI()

# === Load models ===
embedder = SentenceTransformer('BAAI/bge-m3')  # bi-encoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # cross-encoder for reranking

model_name = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Load FAISS index ===
folder_path = "news_articles"
index = faiss.read_index("news_index.faiss")
with open("news_filenames.txt", "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f.readlines()]

# === Define request schema ===
class AskRequest(BaseModel):
    query: str

# === Core QA logic ===
def ask_question(query, k=5, score_threshold=0.3):
    # Bi-encoder embedding and FAISS search
    query_text = "query: " + query
    query_vec = embedder.encode([query_text], normalize_embeddings=True)
    D, I = index.search(query_vec, k=k)

    candidate_pairs = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or score < score_threshold:
            continue
        with open(os.path.join(folder_path, filenames[idx]), "r", encoding="utf-8") as f:
            doc = f.read()
            candidate_pairs.append((query, doc))

    if not candidate_pairs:
        return "❌ 找不到相關新聞，請換一個問題試試。"

    
    # Cross-encoder re-ranking
    scores = cross_encoder.predict(candidate_pairs)
    print(scores)
    reranked = sorted(zip(scores, candidate_pairs), reverse=True)
    top_docs = [doc for _, (_, doc) in reranked[:1]]  # pick top-1 after reranking

    context = "\n\n".join(top_docs)

    # Prompt building
    prompt = f"""請根據以下新聞內容簡短回答問題，只需回答重點，不要重複新聞內容或加額外說明。

            新聞內容：
            {context}

            問題：
            {query}

            回答：
"""

    # Format for LLM (tokenizer-specific formatting)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.2
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()

# === FastAPI endpoint ===
@app.post("/ask")
def ask(req: AskRequest):
    answer = ask_question(req.query)
    return {"query": req.query, "answer": answer}
