---
title: HealthBuddy AI Agent
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🩺 HealthBuddy AI Agent

HealthBuddy is an **agentic AI system** built using **LangGraph + LangChain** that helps users:

- ✅ Ask healthcare-related questions
- 🔎 Research symptoms using **Web Search** and **arXiv**
- 👨‍⚕️ Get doctor recommendations based on symptoms
- 🧠 Uses **ReAct (Reason + Act)** pattern for tool-based reasoning

---

## 🚀 How It Works

1. User enters a health query  
2. AI agent reasons using LLM  
3. Calls tools when needed:
   - Web Search (Tavily)
   - arXiv Research
   - Doctor Recommendation Tool
4. Returns a grounded, structured response

---

## 🧠 Tech Stack

- **Python**
- **LangChain / LangGraph**
- **OpenAI (GPT models)**
- **Gradio** (Web UI)
- **Hugging Face Spaces**

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
