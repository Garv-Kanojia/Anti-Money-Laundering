# app.py
from bedrock_agentcore import BedrockAgentCoreApp
from Multi_Agent_code import app as langgraph_app

# Initialize the AWS AgentCore wrapper
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    """
    This is the function AWS will call when someone sends a request.
    """
    # Extract the prompt from the incoming request payload
    user_prompt = payload.get("prompt", "Analyze these accounts.")
    
    # Set up the state dictionary for LangGraph
    inputs = {
        "prompt": user_prompt,
        "results": []
    }
    
    # Run your multi-agent architecture
    final_state = langgraph_app.invoke(inputs)
    
    # Return the final Suspicious Activity Report (SAR)
    return {"sar_report": final_state["final_sar"]}

if __name__ == "__main__":
    app.run()