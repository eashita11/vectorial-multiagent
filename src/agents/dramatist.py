# src/agents/dramatist.py
from src.retriever import PersonaRetriever
from src.llm import chat

def _last_two(thread):
    """Return (speaker, message) pairs robustly from thread list of tuples or dicts."""
    last = thread[-1]
    prev = thread[-2] if len(thread) > 1 else ("", "")

    def unpack(x):
        if isinstance(x, dict):
            return x.get("speaker", ""), x.get("message", "")
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            return x[0], x[1]
        return "", ""

    def clip(s, n=220):
        s = (s or "").strip().replace("\n", " ")
        return s if len(s) <= n else s[:n] + "…"

    last_spk, last_msg = unpack(last)
    prev_spk, prev_msg = unpack(prev)
    return (last_spk, clip(last_msg)), (prev_spk, clip(prev_msg))

class DramatistAgent:
    def __init__(self, model: str = "llama3"):
        self.retriever = PersonaRetriever("dramatist")
        self.model = model
    

    def respond(self, query: str):
        # Bias retrieval toward stakes/feelings without adding fluff
        aug_query = f"{query} emotions stakes pressure trade-off fear relief motivation consequence"
        hits = self.retriever.search(aug_query, k=3)
        def _useful(h): return len(h["text"].split()) >= 5
        hits = [h for h in hits if _useful(h)][:3]

        context = "\n".join(
            f"- {h['text']} (char {h['character']}, line {h['line_id']}, movie {h['movie_id']})"
            for h in hits
        )

        system = (
            "You are DramatistAgent.\n"
            "Voice: poetic yet precise — you use imagery to reveal emotional truth.\n"
            "Goal (4–5 sentences):\n"
            "- Open with one vivid, cinematic image or metaphor that captures the user's emotional landscape.\n"
            "- Interpret what that image says about their inner tension or desire.\n"
            "- End with a reflective insight or emotional truth that reframes the problem.\n"
            "Rules:\n"
            "- Keep it grounded and sharp, not theatrical.\n"
            "- Speak directly to 'you'; avoid fictional stories or dialogues.\n"
            "- One metaphor max — don’t overdecorate.\n"
            "- Use one citation only if it enriches the emotion or insight.\n"
        )


        user = (
            f"User message:\n{query}\n\n"
            f"Optional grounding snippets:\n{context if context else '(none)'}\n\n"
            "Remember: no fictional names, no story scenes. Speak to *you* (the user)."
        )

        resp_text = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0.2, "num_predict": 200}

        ).strip()

        if len(resp_text) < 30:
            resp_text = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0.2, "num_predict": 384},
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
    def reconcile(self, commander_stmt: str, rationalist_stmt: str) -> dict:
        """
        Brief reconciliation: name the shared core, the live tension, and a next beat.
        3–5 sentences. No quotes or metaphors longer than one sentence.
        """
        system = (
            "You are DramatistAgent.\n"
            "Role: the playwright tying opposing ideas into one emotional arc.\n"
            "Write 3–5 sentences that reconcile Commander and Rationalist:\n"
            "- Begin with an image that captures their contrast.\n"
            "- Reveal their shared heartbeat or motivation beneath the surface.\n"
            "- End with a single insight or call to balance.\n"
            "Keep rhythm and feeling — elegant, not verbose.\n"
        )
        user = f"Commander:\n{commander_stmt}\n\nRationalist:\n{rationalist_stmt}\n\nReconcile briefly:"
        txt = chat(self.model, [{"role":"system","content":system},{"role":"user","content":user}],
                   options={"temperature":0.35}, stream=True).strip()
        if len(txt) < 40:
            txt = chat(self.model, [{"role":"system","content":system},{"role":"user","content":user}],
                       options={"temperature":0.35, "num_predict":384}, stream=False).strip()
        return {"response": txt, "citations": ""}


    def converse(self, query: str, thread: list[dict]) -> dict:
        """
        Respond to the latest point with one vivid but brief image (max one sentence),
        then spell the stakes and add one nudge toward motion. Do NOT repeat Round 1.
        2–4 sentences total. One @mention allowed.
        """
        history = "\n".join(f"{t['speaker']}: {t['message']}" for t in thread[-3:])
        
        system = (
        "You are DramatistAgent.\n"
        "Write 2–4 sentences that feel alive:\n"
        "- Begin with one striking but brief image or line that sets emotional tone.\n"
        "- Then unpack the stakes or tension between the ideas so far.\n"
        "- Close with a line that gently shifts the group toward reflection or harmony.\n"
        "You can @mention one agent if it feels organic.\n"
        "Do NOT repeat Round-1 metaphors — create new texture.\n"
        "Tone: lyrical yet purposeful, emotional but concise.\n"
        )
        
        user = f"User query:\n{query}\n\nRecent thread:\n{history}\n\nYour turn:"
        txt = chat(self.model, [{"role":"system","content":system},{"role":"user","content":user}],
                   options={"temperature":0.35}, stream=False).strip()
        if len(txt) < 30:
            txt = chat(self.model, [{"role":"system","content":system},{"role":"user","content":user}],
                       options={"temperature":0.35, "num_predict":384}, stream=False).strip()
        last_other = next((t["speaker"] for t in reversed(thread)
                   if t.get("speaker") and t["speaker"] != "Dramatist"), None)
        if last_other and "@Dramatist" in txt:
            txt = txt.replace("@Dramatist", f"@{last_other}", 1)
        return {"response": txt, "citations": ""}