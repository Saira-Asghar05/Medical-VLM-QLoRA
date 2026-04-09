# 🩺 End-to-End VLM Fine-Tuning for Medical X-Ray Diagnosis (QLoRA)

**Author:** Saira Asghar  
**Frameworks:** PyTorch, HuggingFace Transformers, PEFT, BitsAndBytes

## 📌 Project Overview
This repository contains a complete, end-to-end pipeline for fine-tuning a Large Vision-Language Model (VLM) on a highly specialized medical dataset. The project demonstrates the ability to adapt a massive 7-Billion parameter model (**LLaVA-1.5-7B**) to interpret chest X-rays and answer diagnostic questions, executed entirely within a severely memory-constrained environment (Google Colab T4 GPU - 15GB VRAM).

## 🚀 Technical Highlights
* **4-bit Quantization (NF4):** Utilized `BitsAndBytesConfig` with double quantization to squeeze the 15GB base model down to ~5GB, completely avoiding CUDA Out-of-Memory (OOM) errors.
* **Parameter-Efficient Fine-Tuning (PEFT):** Injected low-rank adapters ($r=16$) into the `q_proj` and `v_proj` attention matrices, training only ~0.1% of the total model parameters while freezing the base weights.
* **Multi-Modal Data Collation:** Engineered a custom `collate_fn` to process and pad dynamic lengths of image tensors and conversational text from the **VQA-RAD** dataset simultaneously.
* **Inference Optimization:** Resolved catastrophic model degeneration ("greedy loops") during the generation phase by dynamically adjusting context caching, temperature ($0.2$), and repetition penalties ($1.2$).

## 📊 Proof of Concept (Inference Result)
During the evaluation phase on an unseen test X-Ray, the model successfully generated coherent, domain-specific text based on visual cues:

* **Input Query:** "Is there evidence of an aortic aneurysm?"
* **AI Generated Output:** *"Yes, the image shows an aortic aneurysm. It is important to note that this condition requires immediate medical attention..."*

*(Note: The model was trained for a limited number of steps as a Proof of Concept to verify pipeline integrity. Full epoch training on high-compute clusters is required for clinical accuracy).*

## 🛠️ Repository Contents
* `medical_vlm_qlora.ipynb`: The heavily documented Jupyter/Colab notebook containing the entire data ingestion, training, and inference pipeline.

## 🎯 Objective
This project serves as a technical demonstration of advanced AI engineering skills, including handling multi-modal data structures, optimizing LLMs for edge/consumer hardware, and debugging complex Transformer architectures.
