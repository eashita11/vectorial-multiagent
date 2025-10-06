# src/graph/run_graph.py
from __future__ import annotations
from typing import Dict, Any
from src.graph.langgraph_builder import build_graph
from src.graph.state import GraphState

def run_collaboration_graph(query: str, dialogue_rounds: int = 2) -> Dict[str, Any]:
    """
    Execute the LangGraph workflow and return a dict that matches the shape
    expected by the Streamlit UI and/or the legacy orchestrator output.
    """
    graph = build_graph()

    # Seed initial state (init node fills in the rest)
    initial: GraphState = {
        "query": query,
        "dialogue_rounds": int(dialogue_rounds) if dialogue_rounds is not None else 2,
    }

    # Run the graph to completion
    try:
        final_state: GraphState = graph.invoke(initial)  # type: ignore
    except Exception:
        # Ensure we *always* return the expected keys so the UI never crashes
        return {
            "round1": {},
            "dialogue": [],
            "challenges": {},
            "synthesis": {},
        }

    # Shape the return exactly like orchestrator/app expects
    return {
        "round1": final_state.get("round1", {}),
        "dialogue": final_state.get("dialogue", []),
        "challenges": final_state.get("challenges", {}),
        "synthesis": final_state.get("synthesis", {}),
    }