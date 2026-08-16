from langchain_core.tools import tool

@tool
def compute_phq9_score(scores: list[int]) -> str:
    """Calculates PHQ-9 depression severity from exactly 9 item scores (each ranging from 0 to 3)."""
    if len(scores) != 9 or any(s < 0 or s > 3 for s in scores):
        return "Error: PHQ-9 requires exactly 9 scores, each ranging from 0 to 3."
    
    total = sum(scores)
    
    if total <= 4:
        severity = "Minimal / None"
    elif total <= 9:
        severity = "Mild"
    elif total <= 14:
        severity = "Moderate"
    elif total <= 19:
        severity = "Moderately Severe"
    else:
        severity = "Severe"
        
    return f"Total PHQ-9 Score: {total}/27. Severity: {severity}."