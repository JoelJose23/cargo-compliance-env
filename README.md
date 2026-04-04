---
title: Cargo-DeAI-Dispatcher
emoji: 🚢
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: 24.0.5
app_file: app.py
pinned: false
license: apache-2.0
---

# 📦 Cargo Compliance AI Agent (DeAI Dispatcher)

A robust, multi-stage reinforcement learning environment for verifying international trade compliance. This project uses high-parameter LLMs to navigate complex bilateral trade laws between regions like the **EU, US, India, Australia, and Canada**.

## 🚀 The Core Innovation: "The Confidence Trap" Guardrail
Large Language Models are often "confidently wrong." They use professional legal jargon to hide hallucinations. This project implements a **Hard Guardrail System**:

1.  **Phase 1 (Extraction):** Standardizes raw manifest text into a strict schema.
2.  **Phase 2 (Verified Selection):** Matches agent choices against a **ground-truth SQL/JSON registry**. The agent is penalized for using "external knowledge" not found in the verified dataset.
3.  **Phase 3 (Audit):** A high-tier LLM Judge (Llama-3.3-70B) evaluates the reasoning, but only after the mechanical checks in Phase 2 have passed.

## 🛠️ Technical Stack
* **Environment:** Custom `OpenEnv` server implementation.
* **Models Tested:** `Mistral-7B-v0.3`, `Qwen-2.5-72B`, `Kimi-K2-Instruct`.
* **Infrastructure:** Python 3.10+ with `asyncio` and `Groq/HuggingFace` Inference APIs.
* **Hardware Target:** Lenovo Legion 5 Pro (Optimized for local inference via Ollama or remote API).

## 📊 Benchmark Performance
| Metric | Local Mistral | HF Kimi-K2 |
| :--- | :--- | :--- |
| **Extraction Accuracy** | 98% | 100% |
| **Bilateral Law Match** | 85% | 92% |
| **Avg. Reward Score** | 3.2 | 4.17 |

## 🏃 How to Run
1. **Clone the Space:**
   ```bash
   git clone [https://huggingface.co/spaces/your-username/cargo-compliance](https://huggingface.co/spaces/your-username/cargo-compliance)