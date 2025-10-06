# tests/test_retriever.py
from src.retriever import PersonaRetriever

def test_retriever_returns_hits():
    r = PersonaRetriever("commander")
    hits = r.search("conflict", k=3)
    assert isinstance(hits, list)
    assert len(hits) > 0
    assert "text" in hits[0]
    assert "line_id" in hits[0]