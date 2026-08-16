<!-- # 🧠 AI Psychiatric Clinical Intake Agent

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
ollama pull nomic-embed-text # Embedding model for RAG -->

# 🧠 AI Psychiatric Clinical Intake Agent

An advanced, locally-hosted AI agent designed to assist psychiatrists with patient intake, preliminary triaging, and evidence-based treatment drafting. Built with **LangGraph**, **LangChain**, **Ollama**, and **Streamlit**.

## 🏗️ Architecture & Workflow

This project utilizes a **ReAct (Reason + Act)** framework managed by a state machine (LangGraph). It ensures the LLM plans its steps and utilizes clinical tools before communicating with the patient.

1. **Safety Triage Guardrail**: Before the LLM processes any user input, the text is evaluated by a strictly-prompted model (`guardrails.py`). If active crisis, self-harm, or severe psychosis is detected, the workflow instantly bypasses the agent and routes to the **Crisis Escalation Node**, delivering emergency resources.
2. **LangGraph Orchestrator**: The core state machine (`agent_graph.py`) manages short-term conversational memory, maintains the patient's context window, and orchestrates the loop between the LLM and its tools.
3. **Tools & Execution**:
* **RAG Engine** (`rag_engine.py`): Semantic search over local `ChromaDB`.
* **Clinical Calculators** (`clinical_tools.py`): Sandboxed python tools to securely calculate psychiatric scales (e.g., PHQ-9, GAD-7).



---

## 🛠️ Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) installed and running locally.
* [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer). Install it via `pip install uv` if you don't have it.

### Required Local Models

Before running the application, pull the required local models via Ollama in your terminal:

```bash
ollama pull llama3.3:70b     # Foundation LLM (or substitute with mistral-nemo for lower hardware specs)
ollama pull nomic-embed-text # Embedding model for RAG

```

---

## 🚀 Installation & Setup using `uv`

Using `uv` makes dependency installation nearly instant.

1. **Clone the repository and navigate to the directory:**
```bash
git clone <your-repo-url>
cd psych-intake-agent

```


2. **Create a virtual environment and install dependencies:**
```bash
uv venv
# On Windows activate with: .\.venv\Scripts\activate
# On Mac/Linux activate with: source .venv/bin/activate

uv pip install -r requirements.txt

```



---

## ⚠️ CRITICAL: Database Initialization (Do This First!)

Because this project uses a local SQLite-based vector database (ChromaDB), **the database must be populated with at least one document before the chat agent can run.** Furthermore, local ChromaDB does not support simultaneous read/write operations from two different applications.

**Before running the chat app:**

1. Start the Admin App:
```bash
streamlit run admin_app.py --server.port 8502

```


2. Upload your clinical guidelines/PDFs through the Admin UI.
3. Wait for the success message confirming chunks were added to ChromaDB.
4. **Close/Stop the Admin App (Ctrl+C)** to release the database lock.

---

## 💻 Running the Patient Agent

Once the database is initialized and the admin app is closed, you can safely run the main patient-facing chat interface:

```bash
streamlit run app.py

```

*(Terminal logs will display full LangGraph reasoning and execution traces while you chat).*

---

## 🔒 Safety & Privacy Notes

* **Local Execution:** All inference and embedding generation happens locally via Ollama. No Protected Health Information (PHI) is sent to external APIs.
* **Crisis Routing:** The `triage_node` acts as a deterministic circuit breaker for psychiatric emergencies.