import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Step 1: Load Qwen LLM
model_id = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Step 2: Load FAISS index
folder_path = "news_articles"
index = faiss.read_index("news_index.faiss")
with open("news_filenames.txt", "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f.readlines()]

# Step 3: Load embedding model
embedder = SentenceTransformer('BAAI/bge-m3')



def ask_question(query, k=1):
    # Step 1: Embed the query with "query:" prefix (bge-m3 style)
    query_text = "query: " + query
    query_vec = embedder.encode([query_text], normalize_embeddings=True)  # normalize because FAISS is IP

    # Step 2: Search in FAISS index
    D, I = index.search(query_vec, k=k)

    # Step 3: Retrieve top-k documents
    retrieved_texts = []
    for idx in I[0]:
        if idx == -1:
            continue  # If no match found
        with open(os.path.join(folder_path, filenames[idx]), "r", encoding="utf-8") as f:
            retrieved_texts.append(f.read())

    if not retrieved_texts:
        return "❌ 找不到相關新聞，請換一個問題試試。"

    context = "\n\n".join(retrieved_texts)

    print(context)

    # Step 4: Build LLM prompt
    prompt = f"""請根據以下新聞內容簡短回答問題，只需回答重點，不要重複新聞內容或加額外說明。

新聞內容：
{context}

問題：
{query}

回答：
"""

    # Step 5: Format for Qwen model
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

    # Step 6: Generate Answer
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        temperature=0.2
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #print(response)
    return response 

# 🔍 Try it
query = "黃金市況如何？"
#query_text = "query: " + query
answer = ask_question(query)
print("🤖 答案:", answer)
