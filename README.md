# senior-ml-portfolio
# Multi-GPU Vision-Language Model (VLM) Portfolio Project

## Overview
This project demonstrates the design, optimization, and deployment of a Vision-Language Model (VLM) for real-world image understanding tasks. Using the CLIP architecture (ViT + text embedding transformer), the model encodes images and text into a shared embedding space for similarity search and multimodal classification.

Focus areas include **high-performance, scalable AI pipelines**, profiling, and benchmarking — critical skills for deep learning engineering and AI optimization roles.

---

## Dataset

- **Dataset:** Berkeley DeepDrive (BDD100K)  
- **Domain:** Autonomous driving / real-world images  
- **Images:** 10,005 images  
  - Train: 7,005  
  - Validation: 1,000  
  - Test: 2,000  

**Purpose:** Showcase practical ML application on realistic, diverse datasets.

---

## Model Architecture

- **Backbone:** CLIP ViT image encoder + text embedding transformer  
- **Embedding Dimension:** 768  
- **Pretrained Weights:** `openai/clip-vit-base-patch32`  
- **Device Setup:** CPU or GPU (multi-GPU supported via DDP)  
- **Frameworks:** PyTorch, Hugging Face Transformers, ONNX  

---

## Key Features

### Multi-GPU Training Support
- Distributed Data Parallel (DDP) compatible  
- Mixed precision (FP16/BF16) for faster training  

### High-Performance Inference
- Exported model to ONNX  
- Benchmarked latency and throughput for different batch sizes  

### Industry-Relevant Profiling

| Batch Size | Avg Latency (ms) |
|------------|----------------|
| 1          | 179.33         |
| 2          | 324.73         |
| 4          | 615.26         |
| 8          | 1238.16        |
| 16         | 2478.28        |

Optional throughput calculation included for production-grade evaluation.

### Deployment-Ready
- Dockerized API for real-time inference  
- Optional small web interface for demo

---

## How to Use

### 1️⃣ Clone Repository
```bash
git clone https://github.com/amozer1/senior-ml-portfolio.git
cd senior-ml-portfolio/project1_vision_language

