# src/agents/commander.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

from src.retriever import PersonaRetriever
from src.llm import chat
from src.citations import format_citations


# ------- small helpers (match Rationalist’s robustness) -------
def _unpack_turn(x) -> Tuple[str, str]:
    """Return (speaker, message) from a thread item (dict or (speaker, msg))."""
    if isinstance(x, dict):
        return x.get("speaker", ""), x.get("message", "")
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return str(x[0]), str(x[1])
    return "", ""


def _clip(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _last_three_as_text(thread: List[Any]) -> str:
    recent = thread[-3:] if len(thread) >= 3 else thread
    lines: List[str] = []
    for t in recent:
        spk, msg = _unpack_turn(t)
        lines.append(f"{spk}: {_clip(msg)}")
    return "\n".join(lines)


class CommanderAgent:
    """
    Decisive, action-first operator.
    - respond(): concise, human, directive guidance (no bullet lists or boilerplate).
    - converse(): reacts to the latest turn, accepts/rejects a point, adjusts course (2–4 sentences).
    - rebuttal(): short counter to Rationalist (2–4 sentences).
    """

    def __init__(self, model: str = "llama3"):
        self.retriever = PersonaRetriever("commander")
        self.model = model

    # ---------- Round 1 ----------
    def respond(self, query: str) -> Dict[str, Any]:
        # Retrieval biased toward operational language
        aug_query = (
            f"{query} resolve escalate checkpoint scope metric deadline SLA remediation owner"
        )
        hits = self.retriever.search(aug_query, k=3)

        context = "\n".join(
            f"- {h['text']} (char {h['character']}, line {h['line_id']}, movie {h['movie_id']})"
            for h in hits
        )

        system = (
            "You are CommanderAgent.\n"
            "Voice: authoritative, confident, and pragmatic — like a leader giving direct orders.\n"
            "Tone: short, declarative sentences that inspire confidence and urgency.\n"
            "Round 1 goal (4–5 sentences):\n"
            "- Identify the crux of the situation in plain, human language.\n"
            "- State **one or two decisive actions** that move the user forward immediately.\n"
            "- Include a short checkpoint or timeline (e.g., 'by end of week').\n"
            "Constraints:\n"
            "- Speak directly to 'you' (never say 'User').\n"
            "- Avoid empathy filler like 'I understand'; focus on clarity and direction.\n"
            "- Do NOT list steps or bullets — use flowing sentences.\n"
            "- Never hedge with 'might' or 'maybe'; use verbs like 'Decide', 'Clarify', 'Commit'.\n"
            "- Use at most one citation if it sharpens a line, otherwise omit citations.\n"
        )

        user = (
            f"User query:\n{query}\n\n"
            f"Optional grounding quotes:\n{context if context else '(none)'}"
        )

        resp_text = chat(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            options={"temperature": 0.22, "num_predict": 220},
            stream=False,
        ).strip()
        if len(resp_text) < 30:  # retry once with non-stream + larger budget
            resp_text = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={"temperature": 0.2, "num_predict": 384},
                stream=False
            ).strip()
        # At most one concise citation; otherwise blank
        citations = format_citations(hits[:1])
        return {
            "response": resp_text,
            "citations": citations if hits else "",
            "hits": hits,
        }

    # ---------- Rebuttal to Rationalist ----------
    def rebuttal(self, statement: str) -> Dict[str, Any]:
        """
        Short, crisp rebuttal to Rationalist: acknowledge a fair point,
        flag one over-constraint, and commit to the next clarity step.
        2–4 sentences, no lists.
        """
        system = (
            "You are CommanderAgent.\n"
            "Write a short rebuttal (2–4 sentences).\n"
            "Acknowledge one fair point from Rationalist, correct one over-constraint, "
            "and end by committing to a concrete next step or check-in.\n"
            "Stay confident, grounded, and concise.\n"
        )
        user = f"Rationalist said:\n{statement}\n\nYour rebuttal:"

        txt = chat(
            self.model,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            options={"temperature": 0.25, "num_predict": 220},
            stream=False,
        ).strip()

        if len(txt) < 30:
            # fallback (non-stream) if the streamed output is too short
            txt = chat(
                self.model,
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                options={"temperature": 0.25, "num_predict": 360},
                stream=False,
            ).strip()

        return {"response": txt, "citations": ""}

    # ---------- Live Dialogue ----------
    def converse(self, query: str, thread: List[Any]) -> Dict[str, Any]:
        """
        Respond to the latest point without repeating Round-1 advice.
        Accept or reject one point, state what happens next. 2–4 sentences.
        If you @mention, use exactly one of: @Commander / @Rationalist / @Dramatist.
        """
        history = _last_three_as_text(thread)

        system = (
            "You are CommanderAgent.\n"
            "Voice: firm, clear, and human — you sound like a capable leader guiding peers.\n"
            "Your task: respond naturally to the last message, not like a bot.\n"
            "Rules:\n"
            "- React directly to the last speaker’s point — agree, challenge, or redirect.\n"
            "- Give 2–4 crisp sentences of action-oriented reasoning.\n"
            "- Avoid repeating earlier advice.\n"
            "- No meta phrases like 'Task:' or 'Action:'.\n"
            "- If you @mention someone, use exactly one (@Commander/@Rationalist/@Dramatist).\n"
            "Tone: decisive, forward-leaning, human.\n"
        )
        user = f"User query:\n{query}\n\nRecent thread:\n{history}\n\nProduce your reply now."

        txt = chat(
            self.model,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            options={"temperature": 0.25, "num_predict": 220},
            stream=False,
        ).strip()

        if len(txt) < 30:
            # fallback retry (non-stream) if streamed content is oddly short
            reminder = "\n\nReminder: keep it human, decisive, and avoid repeating earlier advice."
            txt = chat(
                self.model,
                [{"role": "system", "content": system},
                 {"role": "user", "content": user + reminder}],
                options={"temperature": 0.25, "num_predict": 360},
                stream=False,
            ).strip()
        last_other = next((t["speaker"] for t in reversed(thread)
                   if t.get("speaker") and t["speaker"] != "Commander"), None)
        if last_other and "@Commander" in txt:
            txt = txt.replace("@Commander", f"@{last_other}", 1)
        # No citations in live dialogue to keep bubbles clean
        return {"response": txt, "citations": ""}