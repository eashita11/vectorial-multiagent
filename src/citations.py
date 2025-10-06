# src/citations.py
from typing import List, Dict

def _norm_line_id(x: str | None) -> str:
    lid = (x or "").strip()
    if not lid:
        return "unknown"
    return lid if lid.startswith("L") else f"L{lid}"

def format_citation(hit: Dict) -> str:
    """
    Prefer retriever-provided '__cite' (e.g., CHARACTER — “Movie Title”, line Lxxxx).
    Fall back gracefully if not present.
    """
    # If retriever already formatted a citation, use it.
    pre = hit.get("__cite")
    if pre:
        return pre

    # Fallbacks (older meta may not have movie title)
    who = (hit.get("character") or "Unknown").strip() or "Unknown"
    lid = _norm_line_id(hit.get("line_id"))
    title = (hit.get("title") or "").strip()
    mid = (hit.get("movie_id") or "unknown").strip()

    if title:
        return f'{who} — “{title}”, line {lid}'
    # final fallback with movie_id
    return f"{who} · line {lid} · movie {mid}"

def format_citations(hits: List[Dict], max_items: int = 2) -> str:
    """
    Join up to `max_items` formatted citations with '; '.
    Return empty string when no hits so UIs can hide the caption cleanly.
    """
    if not hits:
        return ""
    return "; ".join(format_citation(h) for h in hits[:max_items])

def snippet(h: Dict, max_len: int = 160) -> str:
    """
    Compact, single-line quote snippet with ellipsis if needed.
    """
    t = (h.get("text") or "").strip().replace("\n", " ")
    return t if len(t) <= max_len else t[: max_len - 1] + "…"