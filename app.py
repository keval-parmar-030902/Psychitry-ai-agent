import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent_graph import app_graph, logger

st.set_page_config(
    page_title="Psychiatric Clinical Intake AI",
    page_icon="🧠",
    layout="wide"
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Clinical Session Info")
    
    # Generate or retain a unique patient thread ID
    if "patient_id" not in st.session_state:
        st.session_state.patient_id = f"PATIENT-{str(uuid.uuid4())[:8].upper()}"
    
    st.markdown(f"**Current Patient ID:** `{st.session_state.patient_id}`")
    
    st.divider()
    st.markdown("### 🛠️ Connected Tools & Models")
    # st.markdown("- **Model:** `Ollama / llama3.1`")
    st.markdown("- **Model:** `NVIDIA NIM / Llama-3.1-70b-instruct`")
    st.markdown("- **Embeddings:** `nomic-embed-text`")
    st.markdown("- **Safety Gate:** Active Triage Filter")
    st.markdown("- **RAG DB:** ChromaDB (DSM-5 & APA Guidelines)")
    st.markdown("- **Scales:** PHQ-9 / GAD-7 Scorer")
    
    st.divider()
    if st.button("🔄 Reset Conversation / New Patient", use_container_width=True):
        st.session_state.messages = []
        st.session_state.patient_id = f"PATIENT-{str(uuid.uuid4())[:8].upper()}"
        st.rerun()

# --- Main Chat UI ---
st.title("🧠 Psychiatric Intake & Assessment Assistant")
st.caption("AI-assisted clinical intake agent. All suggestions are preliminary and require psychiatrist review.")

# Initialize chat message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello. I'm the clinical intake assistant. How have you been feeling lately, and what symptoms or concerns brought you in today?"}
    ]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Box
if prompt := st.chat_input("Describe your symptoms, feelings, or answer intake questions..."):
    # Append human message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare inputs for LangGraph
    thread_id = st.session_state.patient_id
    config = {"configurable": {"thread_id": thread_id}}
    
    input_data = {
        "messages": [HumanMessage(content=prompt)],
        "patient_id": thread_id
    }

    # Stream execution from LangGraph
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Analyzing and retrieving clinical data..."):
            logger.info(f"\n>>> NEW INTAKE TURN: Patient ID [{thread_id}] <<<")
            
            for event in app_graph.stream(input_data, config=config):
                for node_name, output in event.items():
                    logger.info(f"[GRAPH EVENT] Finished node: '{node_name}'")
                    
                    # Capture final messages from agent or crisis nodes
                    if node_name in ["agent", "crisis"]:
                        last_msg = output["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            full_response = last_msg.content

            if full_response:
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                message_placeholder.markdown("*(Tool processed data without direct reply)*")