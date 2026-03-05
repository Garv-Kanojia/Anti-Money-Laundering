import operator
import json
from typing import TypedDict, Annotated, List
import requests
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import dotenv
dotenv.load_dotenv()

api_key = os.getenv("API_KEY")

# 1. Setup Custom LLM Helper
def call_custom_llm(prompt_text: str) -> str:
    """Wrapper for your custom LLM API to keep agent code clean."""
    response = requests.post(
        url="https://lightning.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "openai/gpt-5-nano",
            "messages": [
                {
                    "role": "user",
                    "content": [{ "type": "text", "text": prompt_text }]
                },
            ],
        })
    )
    return (json.loads(response.content)['choices'][0]['message']['content'])

# State of Graph
class AgentState(TypedDict):
    prompt: str
    plan: List[str]  # The list of agents the planner decides to use
    results: Annotated[List[str], operator.add]  # Aggregates results from parallel agents
    final_sar: str


"""
============================================================================================================================
Planner Agent: Responsible for decomposing the main prompt into sub-tasks and assigning them to specialized agents.
============================================================================================================================
"""


def planner_agent(state: AgentState):
    """Uses ReAct-style thinking to decide which parallel agents to call."""
    
    prompt = f"""
    You are a Planning Agent for financial fraud detection.
    Analyze this request: "{state['prompt']}"
    
    Think step-by-step (ReAct) about which data sources are needed to verify the accounts.
    Available sources: "tavily_agent", "kg_agent", "sanctions_agent", "corporate_agent", "pgsql_agent".
    
    Output your thought process, then output ONLY a valid JSON list of the exact agent names you want to delegate to.
    Example output format:
    Thought: The user provided a company name and a person. I need to check corporate registries and sanctions.
    Delegation: ["corporate_agent", "sanctions_agent"]
    """
    
    response = call_custom_llm(prompt)
    
    # Simple extraction logic: find the list in the LLM's response
    try:
        # Extracts the substring starting with '[' and ending with ']'
        list_start = response.find('[')
        list_end = response.rfind(']') + 1
        plan = json.loads(response[list_start:list_end])
    except Exception as e:
        # Fallback in case the LLM fails to output valid JSON
        plan = ["tavily_agent", "kg_agent", "sanctions_agent", "corporate_agent", "pgsql_agent"]
    return {"plan": plan}


"""
============================================================================================================================
Analyzer Agent: Each agent is specialized in a specific domain (e.g., Tavily for transaction analysis, KG for knowledge graph queries, etc.) and processes its assigned sub-task in parallel.
============================================================================================================================
"""


def analyzer_agent(state: AgentState):
    """Analyzes the aggregated data and writes the Suspicious Activity Report."""
    
    # Combine all results into a single string context
    collected_data = "\n".join(state["results"])
    
    prompt = f"""
    You are an expert Fraud Investigator. Review the requested accounts based on the input: "{state['prompt']}"
    
    Here is the aggregated data from our investigative agents:
    {collected_data}
    
    Analyze this data. Determine if the accounts appear fraudulent. 
    Draft a formal Suspicious Activity Report (SAR) detailing your findings, the evidence, and your final conclusion.
    """
    
    sar_report = call_custom_llm(prompt)
    
    return {"final_sar": sar_report}


"""
============================================================================================================================
Worker Agents: These agents perform the actual data retrieval and analysis based on the planner's delegation. Each agent is designed to handle a specific type of data or analysis relevant to financial fraud detection.
============================================================================================================================
"""


# OpenSanctions Agent
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_sanctions_data(prompt: str):
    """Place your OpenSanctions API call here."""
    return "Add it."

def sanctions_agent(state: AgentState):
    """Fetches sanctions data with retries."""
    try:
        data = fetch_sanctions_data(state["prompt"])
        return {"results": [f"Sanctions Data: {data}"]}
    except Exception as e:
        return {"results": [f"Sanctions Data Failed after retries: {str(e)}"]}

# PostgreSQL Database Agent
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_db_data(prompt: str):
    # Place your pgsql connection and query execution here
    return "Add it"

def pgsql_agent(state: AgentState):
    try:
        data = fetch_db_data(state["prompt"])
        return {"results": [f"Postgres Data: {data}"]}
    except Exception as e:
        return {"results": [f"Postgres DB Failed after retries: {str(e)}"]}

# Tavily Web Search Agent
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_tavily_data(prompt: str):
    return "Add it"

def tavily_agent(state: AgentState):
    """Fetches web search data via Tavily with retries."""
    try:
        data = fetch_tavily_data(state["prompt"])
        return {"results": [f"Tavily Web Search Data: {data}"]}
    except Exception as e:
        return {"results": [f"Tavily Search Failed after retries: {str(e)}"]}

# Knowledge Graph Agent
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_kg_data(prompt: str):
    return "Add it"

def kg_agent(state: AgentState):
    """Fetches knowledge graph relationship data with retries."""
    try:
        data = fetch_kg_data(state["prompt"])
        return {"results": [f"Knowledge Graph Data: {data}"]}
    except Exception as e:
        return {"results": [f"Knowledge Graph Query Failed after retries: {str(e)}"]}

# Corporate Registry Agent
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_corporate_data(prompt: str):
    return "Add it"

def corporate_agent(state: AgentState):
    """Fetches corporate registry data with retries."""
    try:
        data = fetch_corporate_data(state["prompt"])
        return {"results": [f"Corporate Registry Data: {data}"]}
    except Exception as e:
        return {"results": [f"Corporate Registry Failed after retries: {str(e)}"]}


"""
============================================================================================================================
LangGraph Code
============================================================================================================================
"""


# Initialize Graph
workflow = StateGraph(AgentState)

# Add all Nodes
workflow.add_node("planner", planner_agent)
workflow.add_node("tavily_agent", tavily_agent)
workflow.add_node("kg_agent", kg_agent)
workflow.add_node("sanctions_agent", sanctions_agent)
workflow.add_node("corporate_agent", corporate_agent)
workflow.add_node("pgsql_agent", pgsql_agent)
workflow.add_node("analyzer", analyzer_agent)

# Define Entry Point
workflow.set_entry_point("planner")

# Define Dynamic Routing Logic
def route_to_workers(state: AgentState) -> List[str]:
    # Returns the list of agents the planner decided to run
    return state["plan"]

# Add Conditional Edges for Parallel Fan-out
workflow.add_conditional_edges(
    "planner",
    route_to_workers,
    {
        "tavily_agent": "tavily_agent",
        "kg_agent": "kg_agent",
        "sanctions_agent": "sanctions_agent",
        "corporate_agent": "corporate_agent",
        "pgsql_agent": "pgsql_agent"
    }
)

# Add Edges for Fan-in (Aggregation)
worker_nodes = ["tavily_agent", "kg_agent", "sanctions_agent", "corporate_agent", "pgsql_agent"]
for node in worker_nodes:
    workflow.add_edge(node, "analyzer")

# End Graph
workflow.add_edge("analyzer", END)

# Compile
app = workflow.compile()