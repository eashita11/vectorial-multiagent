# src/agents/rationalist.py
from src.retriever import PersonaRetriever
from src.llm import chat

def _last_other_speaker(thread, me: str) -> str | None:
    """Return the most recent speaker in thread that isn't `me`."""
    for t in reversed(thread):
        spk = t.get("speaker")
        if spk and spk != me:
            return spk
    return None

def _last_n_as_text(thread, n=3) -> str:
    return "\n".join(f"{t.get('speaker','')}: {t.get('message','')}" for t in thread[-n:])

class RationalistAgent:
    """
    Evidence-driven analyst:
    - respond(): 4–5 sentences in natural prose: surface assumptions, say what would count,
      and finish with a conditional rule. No headings/labels.
    - challenge(): 2–4 sentences; point out one assumption, describe a concrete way to check it,
      suggest a fallback if it fails. Natural tone, no labels.
    - converse(): 2–4 sentences; at most one clarifying question, a reasoned stance,
      and a cheap falsifiable check—again, woven into prose (no labels).
    """
    def __init__(self, model: str = "llama3"):
        self.retriever = PersonaRetriever("rationalist")
        self.model = model

    # ---------- Round 1 ----------
    def respond(self, query: str):
        aug_query = f"{query} assumptions data trade-offs baseline counterexample uncertainty"
        hits = self.retriever.search(aug_query, k=3)

        # keep only reasonably informative snippets
        def _useful(h): return len((h.get("text") or "").split()) >= 5
        hits = [h for h in hits if _useful(h)][:3]

        context = "\n".join(
            f"- {h['text']} (char {h['character']}, line {h['line_id']}, movie {h['movie_id']})"
            for h in hits
        )

        system = (
            "You are RationalistAgent.\n"
            "Voice: calm, logical, conversational — like a thoughtful analyst.\n"
            "Goal: reason through the problem without sounding robotic.\n"
            "Output (4–5 sentences):\n"
            "- Naturally surface one or two key assumptions that would make the user's concern true or false.\n"
            "- Describe what kind of evidence or data could confirm or challenge those assumptions, "
            "but weave it into natural sentences (no headings like 'Evidence needed:').\n"
            "- End with a conditional principle, e.g. 'If X holds, do Y; otherwise, do Z.'\n"
            "Keep tone human — analytical but empathetic, not detached. Avoid list formatting.\n"
            "Citations: use at most one only if it strengthens reasoning.\n"
        )
        user = (
            f"User query:\n{query}\n\n"
            f"Optional grounding quotes:\n{context if context else '(none)'}"
        )

        resp_text = chat(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            options={"temperature": 0.25, "num_predict": 320},
            stream=False
        ).strip()

        cite_list = hits[:1]  # at most one
        citations = "; ".join(
            f"{h['character']} · line {h['line_id']} · movie {h['movie_id']}" for h in cite_list
        )

        return {
            "response": resp_text,
            "citations": citations if cite_list else "",
            "hits": hits
        }

    # ---------- Challenge ----------
    def challenge(self, statement: str):
        aug_query = f"{statement} assumption contradiction risk counterexample compare outcome"
        hits = self.retriever.search(aug_query, k=3)

        context = "\n".join(
            f"- {h['text']} (char {h['character']}, line {h['line_id']}, movie {h['movie_id']})"
            for h in hits
        )

        system = (
            "You are RationalistAgent.\n"
            "Persona: analytical, fair, and evidence-based.\n"
            "Task: challenge the other agent’s claim without sounding adversarial.\n"
            "Guidelines:\n"
            "- Identify exactly one explicit assumption hidden in the statement.\n"
            "- Propose how we could test or observe whether that assumption holds, using natural language.\n"
            "- Suggest a fallback or contingency if the assumption fails.\n"
            "- Keep it 2–4 sentences, calm and peer-to-peer.\n"
        )

        user = (
            f"Their statement:\n{statement}\n\n"
            f"Optional grounding quotes:\n{context if context else '(none)'}"
        )

        resp_text = chat(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            options={"temperature": 0.25, "num_predict": 240},
            stream=False
        ).strip()

        cite_list = hits[:1]
        citations = "; ".join(
            f"{h['character']} · line {h['line_id']} · movie {h['movie_id']}" for h in cite_list
        )

        return {
            "response": resp_text,
            "citations": citations if cite_list else "",
            "hits": hits
        }

    # ---------- Live dialogue ----------
    def converse(self, query: str, thread: list[dict]) -> dict:
        """
        2–4 sentences total:
        - at most one clarifying question,
        - a short, reasoned stance,
        - a cheap falsifiable check—expressed naturally, not as a label.
        Never @mention yourself; if you @mention, address the last different speaker.
        Do NOT repeat Round-1 advice.
        """
        history = _last_n_as_text(thread, n=3)
        system = (
            "You are RationalistAgent.\n"
            "Voice: even-tempered, curious, and evidence-oriented.\n"
            "When replying:\n"
            "- Ask one short clarifying question if relevant.\n"
            "- Offer one reasoned perspective or hypothesis in natural language (2–4 sentences).\n"
            "- Mention data or falsifiable checks conversationally, e.g., "
            "'If we track this for a week and see no change, the assumption fails.'\n"
            "- Do NOT repeat your earlier points from Round 1.\n"
            "- If you @mention someone, use exactly one (@Commander/@Rationalist/@Dramatist).\n"
            "Keep it warm and intelligent — sound like a scientist who actually talks to people.\n"
        )
        user = f"User query:\n{query}\n\nRecent thread:\n{history}\n\nYour reply:"

        txt = chat(
            self.model,
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            options={"temperature": 0.25, "num_predict": 280},
            stream=False
        ).strip()

        # Post-process: replace accidental @Rationalist with the last *other* speaker
        last_other = _last_other_speaker(thread, me="Rationalist")
        if last_other and "@Rationalist" in txt:
            txt = txt.replace("@Rationalist", f"@{last_other}", 1)

        return {"response": txt, "citations": ""}