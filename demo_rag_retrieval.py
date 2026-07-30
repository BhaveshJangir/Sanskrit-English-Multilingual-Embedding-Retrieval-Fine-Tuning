"""
demo_rag_retrieval.py
--------------------
Interactive Mini RAG / Semantic Retrieval Demo.
Given an English query (or Romanized Sanskrit query), retrieves top-k relevant Sanskrit verses
with similarity scores and English explanations.
"""

import os
import sys
import json
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Sanskrit-English Semantic Search & Mini RAG Demo")
    parser.add_argument("--query", type=str, default="What does Lord Krishna say about performing duty without worrying about results?", help="User query in English or transliterated Sanskrit")
    parser.add_argument("--model_path", type=str, default="output/sanskrit_e5_finetuned", help="Model path or Hugging Face ID")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to retrieve")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, "data", "corpus.json")
    
    if not os.path.exists(corpus_path):
        import prepare_dataset
        prepare_dataset.main()

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    try:
        import torch
        from sentence_transformers import SentenceTransformer, util
        
        model_to_load = args.model_path if os.path.exists(os.path.join(script_dir, args.model_path)) else "intfloat/multilingual-e5-small"
        print(f"🤖 Loading Embedding Model: {model_to_load}...")
        model = SentenceTransformer(model_to_load)
        
        is_e5 = "e5" in model_to_load.lower()
        
        # Encode corpus
        sanskrit_texts = [item["sanskrit"] for item in corpus]
        formatted_passages = [f"passage: {text}" if is_e5 else text for text in sanskrit_texts]
        corpus_embeddings = model.encode(formatted_passages, convert_to_tensor=True)
        
        # Encode user query
        user_query = args.query
        formatted_query = f"query: {user_query}" if is_e5 else user_query
        query_embedding = model.encode(formatted_query, convert_to_tensor=True)
        
        # Cosine Similarity Search
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(scores, k=min(args.top_k, len(corpus)))
        
        print("\n" + "="*70)
        print(f"🔎 SEARCH QUERY: \"{user_query}\"")
        print("="*70 + "\n")
        
        for rank, (score, idx) in enumerate(zip(top_results.values, top_results.indices), 1):
            item = corpus[idx.item()]
            sim_percentage = score.item() * 100
            
            print(f"Rank #{rank} | Similarity Score: {sim_percentage:.2f}% | ID: {item['id']}")
            print(f"📜 Sanskrit Verse (Devanagari):\n   {item['sanskrit']}")
            print(f"🔤 Transliteration:\n   {item['transliteration']}")
            print(f"📖 English Translation:\n   {item['english']}\n")
            print("-" * 70)

    except ImportError:
        print("[INFO] PyTorch or SentenceTransformers is not installed in local environment.")
        print("[INFO] Displaying sample structured retrieval output from corpus:\n")
        print("="*70)
        print(f"SEARCH QUERY: \"{args.query}\"")
        print("="*70 + "\n")
        
        # Match sample
        matched = corpus[0]
        print(f"Rank #1 | Similarity Score: 94.85% | ID: {matched['id']}")
        print(f"Sanskrit Verse (Devanagari):\n   {matched['sanskrit']}")
        print(f"Transliteration:\n   {matched['transliteration']}")
        print(f"English Translation:\n   {matched['english']}\n")

if __name__ == "__main__":
    main()
