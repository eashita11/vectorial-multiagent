# tests/test_graph.py
from src.orchestrator import run_collaboration

def test_graph_end_to_end_smoke(monkeypatch):
    # keep it fast by monkeypatching chat to short strings
    import src.llm as llm_mod
    def stub_chat(*args, **kwargs):
        # emit tiny persona-ish content
        return "ok"
    monkeypatch.setattr(llm_mod, "chat", stub_chat)

    out = run_collaboration("quick smoke", dialogue_rounds=1)
    assert "round1" in out and "dialogue" in out and "challenges" in out and "synthesis" in out
    assert set(out["round1"].keys()) == {"commander","rationalist","dramatist"}