# tests/test_agents.py
from src.agents.commander import CommanderAgent
from src.agents.rationalist import RationalistAgent
from src.agents.dramatist import DramatistAgent

def test_commander_supportive_structure(patch_retriever, patch_llm):
    a = CommanderAgent()
    r = a.respond("handle conflict at work")
    assert "response" in r and "citations" in r and "hits" in r
    assert "GUIDANCE" in r["response"]            # persona signal
    # if retriever returned hits, we expect non-empty or "—"
    assert r["citations"] != "—"

def test_rationalist_questioning_and_challenge(patch_retriever, patch_llm):
    a = RationalistAgent()
    r = a.respond("handle conflict")
    assert "CHALLENGE" in r["response"]
    c = a.challenge("This is a claim to test")
    assert "CHALLENGE" in c["response"]
    assert c["citations"] != "—"

def test_dramatist_evocative(patch_retriever, patch_llm):
    a = DramatistAgent()
    r = a.respond("what happened?")
    assert "SCENE" in r["response"]
    # dramatist shouldn’t sound like advice
    assert "GUIDANCE" not in r["response"]