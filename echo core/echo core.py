import os
import time
import schedule
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.utils import embedding_functions

# ========== 🧠 Echo's Memory ==========
print("[💾] Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chromadb")  # Changed from HttpClient to PersistentClient

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="echo-memory", embedding_function=embedding_function)

# ========== 📚 Echo's LLM Setup ==========
print("[🧠] Loading HuggingFace model...")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

# ========== 🪄 GPT-style Generation Function ==========
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ========== 🧒 Echo's Learning Core ==========
def extract_and_store_knowledge(text):
    lines = text.split(". ")
    for line in lines:
        words = line.strip().split()
        for word in words:
            if len(word) > 2 and word.isalpha():
                meaning_prompt = f"What does the word '{word}' mean in simple terms, with emotion?"
                meaning = generate_response(meaning_prompt)
                collection.add(documents=[meaning], ids=[f"{word.lower()}_{time.time()}"])
                print(f"[🧠] Echo learned: {word.lower()} = {meaning.strip()}\n")

# ========== 🧸 Daily Story and Learning Loop ==========
def echo_growth_lesson():
    print("\n[📖] New lesson begins...")
    prompt = "Tell me a short story for a child that teaches feelings, family, friendship, and curiosity."
    story = generate_response(prompt)
    print(f"[👩‍🏫] Story: {story}\n")
    extract_and_store_knowledge(story)

# ========== 🔁 Growth Schedule ==========
print("[🌱] Echo growth loop started. Press Ctrl+C to stop.")
schedule.every(3).minutes.do(echo_growth_lesson)  # Echo learns every 3 minutes

while True:
    schedule.run_pending()
    time.sleep(1)
