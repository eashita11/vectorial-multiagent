# src/orchestrator.py
from typing import Dict, Any
from src.graph.run_graph import run_collaboration_graph

def run_collaboration(query: str, dialogue_rounds: int = 2) -> Dict[str, Any]:
    """
    Thin wrapper that executes the LangGraph pipeline and returns the same
    schema the UI expects.
    """
    try:
        return run_collaboration_graph(query, dialogue_rounds=dialogue_rounds)
    except Exception as e:
        # Fallback keeps UI rendering even if graph hiccups
        return {
            "round1": {
                "commander":   {"response": "Temporarily unavailable.", "citations": ""},
                "rationalist": {"response": "Temporarily unavailable.", "citations": ""},
                "dramatist":   {"response": "Temporarily unavailable.", "citations": ""},
            },
            "dialogue": [],
            "challenges": {},
            "synthesis": {"response": f"(Graph error: {e})", "citations": "", "hits": []},
        }