# 🧠 AI Psychiatric Clinical Intake Agent

An advanced, locally-hosted AI agent designed to assist psychiatrists with patient intake, preliminary triaging, and evidence-based treatment drafting. Built with **LangGraph**, **LangChain**, **Ollama**, and **Streamlit**.

## 🏗️ Architecture & Workflow

This project utilizes a **ReAct (Reason + Act)** framework managed by a state machine (LangGraph). It ensures the LLM plans its steps and utilizes clinical tools before communicating with the patient.

1. **Safety Triage Guardrail**: Before the LLM processes any user input, the text is evaluated by a strictly-prompted model (`guardrails.py`). If active crisis, self-harm, or severe psychosis is detected, the workflow instantly bypasses the agent and routes to the **Crisis Escalation Node**, delivering emergency resources.
2. **LangGraph Orchestrator**: The core state machine (`agent_graph.py`) that manages short-term conversational memory, maintains the patient's context window, and orchestrates the loop between the LLM and its tools.
3. **Tools & Execution**:
   - **RAG Engine** (`rag_engine.py`): Semantic search over local `ChromaDB`. Queries evidence-based treatments and DSM-5 criteria using `nomic-embed-text` embeddings.
   - **Clinical Calculators** (`clinical_tools.py`): Sandboxed python tools to securely calculate psychiatric scales (e.g., PHQ-9, GAD-7) avoiding LLM math hallucinations.
4. **User Interfaces**:
   - `app.py`: The patient-facing Streamlit chat interface.
   - `admin_app.py`: The admin-facing Streamlit interface for ingesting new clinical PDFs into the RAG database.

## 🛠️ Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.

### Required Local Models
Before running the application, pull the required local models via Ollama:
```bash
ollama pull llama3.3:70b     # Foundation LLM (or substitute with mistral-nemo for lower hardware specs)
ollama pull nomic-embed-text # Embedding model for RAG