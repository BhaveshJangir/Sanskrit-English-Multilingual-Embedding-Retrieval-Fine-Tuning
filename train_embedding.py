"""
train_embedding.py
------------------
Fine-tuning a Multilingual Embedding Model for Sanskrit + English Semantic Search.
Uses SentenceTransformers with MultipleNegativesRankingLoss for contrastive learning.
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"

def load_jsonl(filepath):
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs

def prepare_input_examples(data_pairs, is_e5=True):
    examples = []
    for item in data_pairs:
        anchor = item["anchor"]
        positive = item["positive"]
        
        # E5 models require 'query: ' and 'passage: ' prefixes for optimal performance
        if is_e5:
            anchor_text = f"query: {anchor}"
            positive_text = f"passage: {positive}"
        else:
            anchor_text = anchor
            positive_text = positive
            
        examples.append(InputExample(texts=[anchor_text, positive_text]))
    return examples

def main():
    parser = argparse.ArgumentParser(description="Fine-tune multilingual embedding model for Sanskrit-English retrieval")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Hugging Face model identifier")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.jsonl and test.jsonl")
    parser.add_argument("--output_dir", type=str, default="output/sanskrit_e5_finetuned", help="Directory to save model")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, args.data_dir, "train.jsonl")
    test_path = os.path.join(script_dir, args.data_dir, "test.jsonl")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Run prepare_dataset.py first.")
        
    train_pairs = load_jsonl(train_path)
    test_pairs = load_jsonl(test_path)
    
    is_e5 = "e5" in args.model_name.lower()
    train_examples = prepare_input_examples(train_pairs, is_e5=is_e5)
    
    print(f"🚀 Loading base embedding model: {args.model_name}...")
    model = SentenceTransformer(args.model_name)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Prepare Evaluation set
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for idx, item in enumerate(test_pairs):
        q_id = f"q_{idx}"
        doc_id = f"d_{idx}"
        
        q_text = f"query: {item['anchor']}" if is_e5 else item['anchor']
        doc_text = f"passage: {item['positive']}" if is_e5 else item['positive']
        
        queries[q_id] = q_text
        corpus[doc_id] = doc_text
        relevant_docs[q_id] = {doc_id}
        
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name="sanskrit_eval", score_functions={"cosine": torch.nn.functional.cosine_similarity}
    )
    
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1)
    save_path = os.path.join(script_dir, args.output_dir)
    
    print(f"🏋️ Starting Fine-Tuning ({args.epochs} Epochs, Batch Size: {args.batch_size}, LR: {args.lr})...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=save_path,
        optimizer_params={'lr': args.lr},
        show_progress_bar=True
    )
    
    print(f"✅ Training completed! Fine-tuned model saved to: {save_path}")

if __name__ == "__main__":
    main()
