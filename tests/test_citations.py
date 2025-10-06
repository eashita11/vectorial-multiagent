# tests/test_citations.py
from src.agents.commander import CommanderAgent
import src.agents.commander as commander_mod

def test_citations_present_when_hits(monkeypatch):
    a = CommanderAgent()
    # force one hit
    class Ret:
        def search(self, q, k=3, initial=50, lambda_mmr=0.7):
            return [{"line_id":"L9","movie_id":"m9","character":"DEL","text":"Counterpoint evidence."}]
    a.retriever = Ret()

    # stub model
    monkeypatch.setattr(commander_mod.chat,
        "__call__", lambda *args, **kwargs: "GUIDANCE: grounded answer."
    )
    # monkeypatching bound function above is clunky; better:
    monkeypatch.setattr(commander_mod, "chat",
        lambda *args, **kwargs: "GUIDANCE: grounded answer."
    )

    r = a.respond("query")
    assert r["citations"] != "—"
    assert "DEL" in r["citations"] and "L9" in r["citations"]

def test_no_forced_citation_when_no_hits(monkeypatch):
    a = CommanderAgent()
    class Empty:
        def search(self, q, k=3, initial=50, lambda_mmr=0.7): return []
    a.retriever = Empty()

    monkeypatch.setattr(commander_mod, "chat",
        lambda *args, **kwargs: "GUIDANCE: limited evidence, proceed carefully."
    )

    r = a.respond("query")
    assert r["response"]
    assert r["citations"] in ("—", "")   