# src/agents/synthesizer.py
from src.llm import chat

def _strip_fences(text: str) -> str:
    """Remove accidental code fences or JSON wrappers."""
    t = (text or "").strip()
    # Strip markdown fences
    if t.startswith("```"):
        # remove first fence
        t = t.split("```", 1)[-1].strip()
        # if a closing fence remains, drop everything after it
        if "```" in t:
            t = t.split("```", 1)[0].strip()
    # If the model returned a JSON object by mistake, try to pull a 'response' field
    if t.startswith("{") and '"response"' in t:
        try:
            import json
            j = json.loads(t)
            candidate = (j.get("response") or "").strip()
            if candidate:
                return candidate
        except Exception:
            pass
    return t

class SynthesizerAgent:
    """
    Reads multiple persona responses (already grounded/cited) and produces a concise,
    balanced synthesis as a single prose answer with 3 numbered next steps.
    """

    def __init__(self, model: str = "llama3"):
        self.model = model

    def synthesize(self, query: str, persona_msgs: list[tuple[str, dict]]) -> dict:
        """
        persona_msgs: list of tuples (speaker_name, message_dict)
          message_dict keys: "response", "citations", "hits"
        """
        # Build structured context the model can rely on (don’t let it invent sources)
        blocks = []
        for speaker, msg in persona_msgs:
            resp = (msg.get("response") or "").strip()
            cits = (msg.get("citations") or "").strip()
            blocks.append(
                f"{speaker.upper()}:\n"
                f"{resp}\n"
                f"Provided citations: {cits if cits else '(none)'}"
            )
        context = "\n\n".join(blocks)

        system = (
            "You are SynthesizerAgent.\n"
            "Your role: blend Commander’s clarity, Rationalist’s logic, and Dramatist’s emotional depth "
            "into one unified voice.\n"
            "Write exactly 5–6 sentences (≤150 words):\n"
            "- Open with a crisp summary of what the three agree on.\n"
            "- Compare and merge their perspectives naturally, naming them explicitly.\n"
            "- Capture the shared insight in one strong, human sentence.\n"
            "- End with 'Next steps:' and 2–3 short imperatives.\n"
            "Tone: mature, reflective, and complete — ensure the last line finishes fully.\n"
            "Citations: reuse persona citations verbatim if relevant; never invent new ones.\n"
        )

        user = (
            f"User query:\n{query}\n\n"
            f"Persona outputs (do NOT alter their wording; only reference them):\n{context}\n\n"
            "Produce the final synthesis now."
        )

        resp_text = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": 0.2, "num_predict": 256},
            stream=False,
        ).strip()

        # Post-process to avoid JSON / code blocks and keep only clean prose
        resp_text = _strip_fences(resp_text)
        if not resp_text or resp_text.startswith("{"):
            # Fallback non-streaming with a stronger reminder
            user_fallback = user + "\n\nReminder: return plain prose, not JSON or code blocks."
            resp_text = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_fallback},
                ],
                options={"temperature": 0.2, "num_predict": 256},
                stream=False,
            ).strip()
            resp_text = _strip_fences(resp_text)

        return {
            "agent": "synthesis",
            "response": resp_text,
            "citations": "inline (reused from personas when present)",
            "hits": []
        }