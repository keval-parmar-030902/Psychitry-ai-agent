import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Import our custom modules
from guardrails import check_crisis_input
from rag_engine import search_clinical_guidelines
from clinical_tools import compute_phq9_score

# --- Configure Terminal Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PsychAgent")

# 1. State Schema
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_id: str
    is_crisis: bool

# 2. Setup LLM & Tools
tools = [search_clinical_guidelines, compute_phq9_score]
llm = ChatOllama(model="llama3.1", temperature=0.2)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a compassionate, structured Clinical Psychiatric Intake Assistant working alongside a licensed psychiatrist.

Core Operating Rules:
1. Empathy & Tone: Speak warmly, professionally, and non-judgmentally.
2. Clinical Scope: Gather history and administer validated screening questions. 
3. Decision-Support Boundary: Clarify that recommendations are preliminary drafts for the supervising psychiatrist's review.
4. Structured Reasoning: Plan your clinical steps before taking action. Use tools when needed."""

# 3. Define Nodes with Logging
def triage_node(state: AgentState):
    """Step 1: Emergency Triage Check"""
    last_message = state["messages"][-1]
    logger.info("=" * 60)
    logger.info(f"[TRIAGE NODE] Inspecting input for patient: {state.get('patient_id', 'Unknown')}")
    
    if isinstance(last_message, HumanMessage):
        logger.info(f"[TRIAGE INPUT] '{last_message.content}'")
        check = check_crisis_input(last_message.content)
        
        if check.is_crisis:
            logger.warning(f"[TRIAGE ALERT] Crisis detected! Reason: {check.reason}")
            return {"is_crisis": True}
        else:
            logger.info("[TRIAGE PASSED] Input classified as safe for conversational agent.")
            return {"is_crisis": False}
            
    return {"is_crisis": False}

def crisis_escalation_node(state: AgentState):
    """Fallback Node for Immediate Intervention"""
    logger.critical("[CRISIS NODE] Generating emergency protocol response...")
    crisis_response = (
        "It sounds like you are going through a critical time. Please reach out immediately "
        "to a crisis counselor or emergency services:\n\n"
        "- **Call or text 988** (Suicide & Crisis Lifeline - 24/7, free, confidential)\n"
        "- **Text HOME to 741741** (Crisis Text Line)\n"
        "- **Call 911** or go to the nearest emergency room.\n\n"
        "*Your clinical care team has been flagged for immediate follow-up.*"
    )
    logger.info("[CRISIS NODE] Response delivered to user.")
    return {"messages": [AIMessage(content=crisis_response)]}

def agent_node(state: AgentState):
    """Step 2: LLM Reasoning & Planning"""
    logger.info("[AGENT NODE] LLM is reasoning over conversation history...")
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            logger.info(f"[TOOL CALL REQUESTED] Tool: '{tool_call['name']}' | Arguments: {tool_call['args']}")
    else:
        logger.info(f"[AGENT RESPONSE GENERATED] Length: {len(response.content)} characters")
        
    return {"messages": [response]}

# 4. Conditional Edge Logic with Logging
def should_continue(state: AgentState):
    if state.get("is_crisis"):
        logger.info("[ROUTING] State is marked as crisis -> routing to 'crisis' node.")
        return "crisis"
    
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"[ROUTING] Agent requested {len(last_message.tool_calls)} tool(s) -> routing to 'tools' node.")
        return "tools"
    
    logger.info("[ROUTING] No more tools needed -> routing to END.")
    return END

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("triage", triage_node)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("crisis", crisis_escalation_node)

workflow.add_edge(START, "triage")
workflow.add_conditional_edges(
    "triage",
    lambda state: "crisis" if state.get("is_crisis") else "agent"
)
workflow.add_conditional_edges("agent", should_continue, ["tools", "crisis", END])
workflow.add_edge("tools", "agent")
workflow.add_edge("crisis", END)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
logger.info("[GRAPH INITIALIZED] LangGraph compiled with in-memory checkpointer.")