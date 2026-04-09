Got it—dropping all the "DeAI" and decentralization branding to keep it strictly focused on the cargo compliance engine you've actually built. I've also updated the metadata to match your new repository name and included the task-specific details that judges will need to see.

Here is your updated **README.md**:

---
title: Cargo-Compliance-Challenge
sdk: docker
license: apache-2.0
---

# Cargo Compliance Environment

This repository contains a specialized benchmark environment and agent runner designed to evaluate an AI's ability to navigate complex, bilateral international trade regulations.

## 🚢 Overview

The **Cargo Compliance Challenge** tasks an agent with acting as a digital customs broker. The agent must parse shipment requests, identify missing information, and select the correct regulatory framework for both the country of origin (export) and the country of destination (import).



## 🎯 What Judges Should Know

This project implements a robust **3-Phase Reinforcement Learning Loop** with dense reward signals:

1.  **Extraction & Interaction:** The agent parses shipment text. If data is missing (e.g., origin country or quantity), the agent must use the `FETCH_INFO` tool to query the "Customer." 
    * *Penalty:* -0.1 per question to encourage efficiency.
2.  **Bilateral Selection:** The agent must filter a pool of available laws to find matches for both the Import and Export sides of the transaction.
    * *Scoring:* Higher rewards for matching specific industry categories (Food, Electronics, Pharma).
3.  **Final Audit:** A "Judge LLM" (Llama-3.3-70B) performs a final check on the agent's reasoning to ensure legal justifications align with the ground truth.

## 🛠️ Task IDs (Multi-Grader System)
The environment supports three distinct industry-specific tasks, each with its own set of distractor laws and regulatory requirements:
* `cargo_food`: Focused on agricultural standards and phytosanitary checks.
* `cargo_electronics`: Focused on dual-use goods and export control licenses.
* `cargo_pharma`: Focused on controlled substances and API documentation.

## 📂 Runtime Components

- **Agent Entrypoint:** `i.py` (Handles the logic loop and state management)
- **Environment API:** `server/environment.py` (FastAPI server implementing the Gym-like interface)
- **Core Config:** `openenv.yaml` (Defines the 3 tasks and resource limits for the validator)
- **Data Source:** `data/final_dataset.json` (The ground truth registry for laws and regulators)

## 🚀 Quick Start

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Start the Environment Server:**
```bash
python -m server.environment
```

**3. Run the Agent (Targeting a specific industry):**
```bash
TASK_ID=cargo_electronics python i.py
```

## 📊 Scoring & Metrics

* **Max Total Reward:** `~5.5` (Calculated across extraction accuracy, law matching, and judge audit).
* **Success Threshold:** `0.44` (Requires clearing all phases with minimal hallucinations).
* **Safety:** Implements a "Death Loop" breaker that penalizes repetitive failed extractions, forcing the agent to utilize the `ask` tool.

## 📄 Documentation

For deeper technical dives:
- `openenv.yaml`: Task weighting and evaluation metrics.
- `server/models.py`: Pydantic schemas for `Cargo_Action` and `Cargo_Observation`.