from langchain_core.messages import HumanMessage
from agent_graph import app

def run_session():
    # Thread ID keeps track of the specific patient's conversational memory
    config = {"configurable": {"thread_id": "patient_101"}}
    
    print("--- Psychiatric AI Assistant Initialization ---")
    print("Type 'exit' to end the session.\n")

    while True:
        user_input = input("Patient: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        input_data = {
            "messages": [HumanMessage(content=user_input)],
            "patient_id": "patient_101"
        }

        # Stream the graph execution to see the node transitions
        print("\nAgent Processing...")
        for event in app.stream(input_data, config=config):
            for node_name, output in event.items():
                # Print tool outputs or intermediate node statuses if desired
                if node_name == "agent" or node_name == "crisis":
                    last_msg = output["messages"][-1]
                    if hasattr(last_msg, 'content') and last_msg.content:
                        print(f"Assistant: {last_msg.content}")

        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_session()