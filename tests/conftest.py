# tests/conftest.py
import pytest
from unittest.mock import patch

FAKE_HITS = [
    {"line_id": "L1", "movie_id": "m1", "character": "IAN", "text": "Handle conflict directly with calm."},
    {"line_id": "L2", "movie_id": "m2", "character": "RICKY", "text": "Ask clarifying questions first."},
    {"line_id": "L3", "movie_id": "m3", "character": "MARTINS", "text": "Describe the situation neutrally."},
]

@pytest.fixture(autouse=True)
def mock_llm_and_retriever():
    # 1) Mock all LLM calls (no real Ollama calls during tests)
    with patch("src.llm.chat", return_value="(mocked response)"):
        # 2) Mock retriever.search so tests don’t touch FAISS/embeddings
        with patch("src.retriever.PersonaRetriever.search", return_value=FAKE_HITS):
            yield

# --- Shared fake FAISS hits we can reuse across tests ---
@pytest.fixture
def fake_hits():
    return [
        {"line_id": "L1", "movie_id": "m1", "character": "IAN",     "text": "Handle conflict directly with calm."},
        {"line_id": "L2", "movie_id": "m2", "character": "RICKY",   "text": "Ask clarifying questions first."},
        {"line_id": "L3", "movie_id": "m3", "character": "MARTINS", "text": "Describe the situation neutrally."},
    ]

@pytest.fixture
def patch_retriever(monkeypatch, fake_hits):
    # Patch PersonaRetriever.search to return our controlled hits
    from src import retriever as retriever_mod
    def fake_search(self, q, k=3, initial=50, lambda_mmr=0.7):
        return fake_hits[:k]
    monkeypatch.setattr(retriever_mod.PersonaRetriever, "search", fake_search)

@pytest.fixture
def patch_llm(monkeypatch):
    """
    Patch src.llm.chat so unit tests don't call a real server.
    Returns short persona-typed strings based on the system prompt.
    """
    import src.llm as llm_mod

    def persona_stub(model: str, messages: list[dict], options=None, timeout=90, max_retries=0, stream=False):
        sys_msg = (messages[0].get("content") or "").lower() if messages else ""
        if "you are commanderagent" in sys_msg:
            return "GUIDANCE: stay calm; lead; provide direction."
        if "you are rationalistagent" in sys_msg:
            return "CHALLENGE: test logic; find assumptions; weigh evidence."
        if "you are dramatistagent" in sys_msg:
            return "SCENE: concise image; stakes; nudge."
        if "you are synthesizeragent" in sys_msg:
            return "SYNTH: combine leadership + reasoning + narrative [IAN, L1, movie m1]."
        return "OK"

    monkeypatch.setattr(llm_mod, "chat", persona_stub)