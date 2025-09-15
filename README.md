# Emach-Structures-ML-Portfolio
# Multi-GPU Vision-Language Model (VLM) Portfolio Project

# Emach Structures ML Portfolio – Colab Edition

## Overview
This portfolio demonstrates a full-stack Machine Learning workflow for **vision-language models (VLMs), high-performance computer vision (CV), and LLM deployment** using Google Colab. The project spans six phases, combining image processing, defect detection, text-image retrieval, and transformer-based LLM inference.

---

## Phase 1 – Setup & Dataset Preparation
- Mounted **Google Drive** for dataset access.  
- Organized train/validation/test datasets.  
- Verified and sampled images using **PIL** and **matplotlib**.  
- Ensured reproducibility for synthetic datasets and image sampling.  

**Key libraries:** `os`, `PIL`, `matplotlib`, `random`.

---

## Phase 2 – Vision-Language Model (CLIP)
- Installed and loaded **OpenAI CLIP (ViT-B/32)**.  
- Randomly sampled images and encoded them using `CLIPProcessor`.  
- Generated **image embeddings** of shape `[batch_size, 512]`.  
- Exported **CLIP image encoder to ONNX** for high-performance inference.  
- Benchmarked latency across batch sizes.  

**Outputs:**  
- ONNX model saved to Google Drive.  
- Example image embeddings: `[4, 512]`.  

**Key libraries:** `transformers`, `torch`, `onnx`, `onnxruntime`.

---

## Phase 3 – High-Performance CV
- Setup **multi-GPU training & inference profiling** (Colab GPU / TPU).  
- Integrated **YOLOv8** models for defect detection.  
- Performed **dynamic batch inference**, visualization, and risk scoring.  
- Exported YOLO models to ONNX for cross-platform deployment.  

**Outputs:**  
- Latency per batch metrics.  
- Annotated defect detection images.  

**Key libraries:** `ultralytics`, `torchvision`, `onnxruntime`.

---

## Phase 4 – LLM / Transformer Deployment
- Deployed **GPT-2 model** to ONNX for fast CPU/GPU inference.  
- Verified dynamic batching support and sequence length handling.  
- Benchmarked latency across batch sizes.  
- Optional visualization: latency vs batch size.  

**Outputs:**  
- ONNX model for GPT-2 saved.  
- Inference shape: `[batch_size, seq_len, vocab_size]`.  

**Key libraries:** `transformers`, `torch`, `onnx`, `onnxruntime`.

---

## Phase 5 – Colab Integration & Retrieval
- Integrated **image retrieval pipeline** using CLIP embeddings.  
- Fetched sample images and computed similarity scores.  
- Demonstrated ranking for top-4 similar images per query.  

**Outputs:**  
- Example similarity scores:

- All sample images successfully processed.  

**Key libraries:** `transformers`, `torch`, `PIL`, `matplotlib`.

---

## Phase 6 – Cross-Phase FAISS & Search
- Computed **image embeddings** for all phases.  
- Built a **FAISS index** for vector similarity search.  
- Performed **text-image retrieval** with cosine similarity.  
- Evaluated retrieval results across dataset.  
- Generated **summary tables & reports**.  

**Outputs:**  
- FAISS index file saved.  
- Latency and similarity tables.  
- Retrieval demo results for sample images.  

**Key libraries:** `faiss`, `torch`, `numpy`, `matplotlib`.

---

## Final Notes
- All models, embeddings, and ONNX files are stored in **Google Drive** under the folder:
- All sample images successfully processed.  

**Key libraries:** `transformers`, `torch`, `PIL`, `matplotlib`.

---

## Phase 6 – Cross-Phase FAISS & Search
- Computed **image embeddings** for all phases.  
- Built a **FAISS index** for vector similarity search.  
- Performed **text-image retrieval** with cosine similarity.  
- Evaluated retrieval results across dataset.  
- Generated **summary tables & reports**.  

**Outputs:**  
- FAISS index file saved.  
- Latency and similarity tables.  
- Retrieval demo results for sample images.  

**Key libraries:** `faiss`, `torch`, `numpy`, `matplotlib`.

---

## Final Notes
- All models, embeddings, and ONNX files are stored in **Google Drive** under the folder:

- Colab runtime supports multi-GPU and CPU benchmarking.  
- The pipeline is modular: you can **retrain YOLO**, **recompute embeddings**, or **update retrieval** without changing the core structure.
