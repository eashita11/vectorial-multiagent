# tests/test_persona_compliance.py
from src.agents.commander import CommanderAgent
from src.agents.rationalist import RationalistAgent
from src.agents.dramatist import DramatistAgent

def _score_persona(text: str, cues: list[str]) -> int:
    t = text.lower()
    return sum(1 for c in cues if c in t)

def test_persona_compliance_heuristics(patch_retriever, patch_llm):
    m = CommanderAgent().respond("conflict")
    s = RationalistAgent().respond("conflict")
    n = DramatistAgent().respond("conflict")

    commander_score  = _score_persona(m["response"], ["guidance"])
    rationalist_score = _score_persona(s["response"], ["challenge"])
    dramatist_score   = _score_persona(n["response"], ["scene"])

    assert commander_score  >= 1
    assert rationalist_score >= 1
    assert dramatist_score   >= 1