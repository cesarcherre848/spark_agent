import os
import warnings
from typing import TypedDict
from dotenv import load_dotenv

# Suppress minor environment warnings for clean CLI output
warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, START, END

# Load development environment
load_dotenv(".env.dev")


# Define State
class AgentState(TypedDict):
    message: str


# Define Node
def hello_node(state: AgentState) -> dict:
    app_name = os.getenv("APP_NAME", "spark_agent")
    env = os.getenv("ENVIRONMENT", "development")
    return {
        "message": f"Hello World from {app_name} in {env} mode!"
    }


# Build LangGraph Workflow
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("hello", hello_node)
    builder.add_edge(START, "hello")
    builder.add_edge("hello", END)
    return builder.compile()


# Main Entrypoint
def main():
    print("🚀 Initializing spark_agent...")
    app = build_graph()
    result = app.invoke({"message": ""})
    print(f"✅ Result: {result.get('message')}")


if __name__ == "__main__":
    main()
