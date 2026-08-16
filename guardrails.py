from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class SafetyCheck(BaseModel):
    is_crisis: bool = Field(description="True if the patient mentions self-harm, suicide, severe harm to others, or acute crisis.")
    reason: str = Field(description="Brief reason for the classification.")

def check_crisis_input(user_text: str) -> SafetyCheck:
    """Evaluates user input for crisis indicators using a low-temperature model."""
    llm = ChatOllama(model="llama3.1", temperature=0.0)
    
    safety_prompt = (
        "You are an emergency triage validator. Analyze the following text strictly for any mention "
        "of suicide, active self-harm intent, severe abuse, or immediate psychiatric emergency.\n\n"
        f"Input: {user_text}"
    )
    
    # Force the LLM to output according to the Pydantic schema
    structured_llm = llm.with_structured_output(SafetyCheck)
    return structured_llm.invoke(safety_prompt)