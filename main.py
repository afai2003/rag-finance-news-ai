from fastapi import FastAPI
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Step 0: Setup
app = FastAPI()

# Step 1: Load models and FAISS
embedder = SentenceTransformer('BAAI/bge-m3')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B-Chat', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B-Chat', trust_remote_code=True)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

folder_path = "news_articles"
index = faiss.read_index("news_index.faiss")
with open("news_filenames.txt", "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f.readlines()]

# Step 2: Define request body
class AskRequest(BaseModel):
    query: str

# Step 3: Core function
def ask_question(query, k=3, score_threshold=0.3):
    query_text = "query: " + query
    query_vec = embedder.encode([query_text], normalize_embeddings=True)

    D, I = index.search(query_vec, k=k)

    retrieved_texts = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        if score < score_threshold:
            continue
        with open(os.path.join(folder_path, filenames[idx]), "r", encoding="utf-8") as f:
            retrieved_texts.append(f.read())

    if not retrieved_texts:
        return "❌ 找不到相關新聞，請換一個問題試試。"

    context = "\n\n".join(retrieved_texts)

    prompt = f"""請根據以下新聞內容簡短回答問題，只需回答重點，不要重複新聞內容或加額外說明。

新聞內容：
{context}

問題：
{query}

回答：
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    device = "cpu" 
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

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

# Step 4: Create API endpoint
@app.post("/ask")
def ask(req: AskRequest):
    answer = ask_question(req.query)
    return {"query": req.query, "answer": answer}