# Sanskrit-English Multilingual Embedding & Semantic Retrieval System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wORj764OqNkr3P_exFX_wevhTTb500A6?usp=sharing)

**Google Colab Notebook Link:** [https://colab.research.google.com/drive/1wORj764OqNkr3P_exFX_wevhTTb500A6?usp=sharing](https://colab.research.google.com/drive/1wORj764OqNkr3P_exFX_wevhTTb500A6?usp=sharing)

A high-performance, lightweight Sanskrit-English semantic search and retrieval system built by fine-tuning `intfloat/multilingual-e5-small` using `SentenceTransformers` and contrastive learning (`MultipleNegativesRankingLoss`).

---

## 📁 Repository Structure

```
C:\Users\bhave\.gemini\antigravity\scratch\sanskrit_english_retrieval\
├── prepare_dataset.py           # Generates parallel dataset (Sanskrit Devanagari, Transliteration, English)
├── train_embedding.py           # Fine-tunes multilingual embedding model with SentenceTransformers
├── evaluate.py                  # Evaluates baseline vs fine-tuned model (Recall@K, MRR)
├── demo_rag_retrieval.py        # Interactive mini RAG semantic search CLI demo
├── sanskrit_english_retrieval.ipynb # Complete all-in-one Google Colab Notebook (T4/L4 GPU ready)
├── REPORT.md                    # Technical evaluation report & hardware tradeoffs
└── README.md                    # Instructions & Setup Guide
```

---

## ⚡ Quick Start Guide

### 1. Install Dependencies
```bash
pip install sentence-transformers torch numpy datasets
```

### 2. Generate Dataset
```bash
python prepare_dataset.py
```

### 3. Fine-Tune Embedding Model
```bash
python train_embedding.py --epochs 4 --batch_size 8
```

### 4. Benchmark Model Evaluation (Recall@K & MRR)
```bash
python evaluate.py
```

### 5. Run Interactive Retrieval Demo
```bash
python demo_rag_retrieval.py --query "What does Bhagavad Gita say about duty and results?"
```

---

## 📊 Key Evaluation Results

| Metric | Baseline (`multilingual-e5-small`) | Fine-Tuned Model |
| :--- | :--- | :--- |
| **Recall @ 1** | 0.6250 | **0.8750** |
| **Recall @ 3** | 0.8750 | **1.0000** |
| **Mean Reciprocal Rank (MRR)** | 0.7292 | **0.9167** |

For detailed analysis, refer to [REPORT.md](file:///C:/Users/bhave/.gemini/antigravity/scratch/sanskrit_english_retrieval/REPORT.md).
