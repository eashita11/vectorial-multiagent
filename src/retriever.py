# src/retriever.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- simple heuristics --------------------------------------------------------
def _length_ok(t: str, min_tokens=5, max_chars=220) -> bool:
    t = (t or "").strip()
    return len(t.split()) >= min_tokens and len(t) <= max_chars

def _informative(t: str) -> bool:
    # avoid one-word interjections / empties
    bad = {"yes", "no", "what", "well?", "okay?", "ok?", "okay", "huh", "uh"}
    tok = (t or "").lower().strip()
    return len(tok) > 3 and tok not in bad

def _mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_mult=0.7, k=3):
    """
    Simple MMR on unit-normalized vectors (cosine via dot).
    query_vec: (d,), cand_vecs: (n,d) both normalized.
    """
    selected = []
    n = cand_vecs.shape[0]
    if n == 0:
        return selected
    candidates = list(range(n))
    sims = cand_vecs @ query_vec  # (n,)

    while candidates and len(selected) < k:
        if not selected:
            # pick the best similarity among remaining
            best_idx = max(candidates, key=lambda ii: sims[ii])
            candidates.remove(best_idx)
            selected.append(best_idx)
            continue

        def score(ii):
            # penalize redundancy vs. already selected
            max_sim_sel = max(cand_vecs[ii] @ cand_vecs[s] for s in selected)
            return lambda_mult * sims[ii] - (1 - lambda_mult) * max_sim_sel

        best_idx = max(candidates, key=score)
        candidates.remove(best_idx)
        selected.append(best_idx)

    return selected

# ---- retriever ----------------------------------------------------------------
class PersonaRetriever:
    """
    Persona-scoped retriever.
    Looks for:
      data/processed/personas/{persona}/{persona}.meta.jsonl   (required)
      data/processed/personas/{persona}/{persona}.faiss        (optional)
    Falls back gracefully (and quickly) if FAISS or SBERT is unavailable.
    """

    def __init__(self, persona_name: str, base: str = "data/processed/personas"):
        self.name = persona_name
        self.model = None               # lazy SBERT
        self.index = None               # FAISS (optional)
        self.meta = []                  # list of dicts
        self._sbert_loaded = False

        faiss_path = f"{base}/{persona_name}/{persona_name}.faiss"
        meta_path  = f"{base}/{persona_name}/{persona_name}.meta.jsonl"
        # accept legacy filename convention if present
        if not os.path.exists(meta_path):
            legacy = f"{base}/{persona_name}/{persona_name}_meta.jsonl"
            if os.path.exists(legacy):
                meta_path = legacy

        # load meta (required for any retrieval)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self.meta.append(json.loads(line))
                    except Exception:
                        # tolerate a bad line rather than failing entirely
                        continue

        # load FAISS index if present
        if os.path.exists(faiss_path):
            try:
                self.index = faiss.read_index(faiss_path)
            except Exception:
                # keep running without FAISS
                self.index = None

    # --- internals -------------------------------------------------------------

    def _load_model(self):
        """Lazy SBERT init, with optional disable switch for CI."""
        if self._sbert_loaded:
            return
        if os.getenv("RETRIEVER_DISABLE_SBER T", "").strip():  # allow fast CI
            return
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._sbert_loaded = True
        except Exception:
            # If model fails to load, we will fall back to naive top-k
            self.model = None
            self._sbert_loaded = False

    def _filter_meta(self):
        """Apply quick value filters to meta."""
        out = []
        for h in self.meta:
            txt = (h.get("text") or "").strip()
            if _length_ok(txt) and _informative(txt):
                out.append(h)
        return out

    def _naive_return(self, k: int):
        """Fast path: no embeddings — just return the first k filtered rows."""
        filtered = self._filter_meta()
        return filtered[:k]

    # --- public ---------------------------------------------------------------

    def search(self, query: str, k: int = 3, initial: int = 50, lambda_mmr: float = 0.7):
        """
        Return up to k diverse, informative lines for this persona.
        - If FAISS index is available, do ANN -> filter -> re-embed -> MMR
        - Else, try SBERT-only MMR on a capped subset
        - Else, naive top-k filtered
        """
        if not self.meta:
            return []

        # FAISS path
        if self.index is not None:
            self._load_model()
            if not self.model:
                # If SBERT didn't load, at least return naive filtered
                return self._naive_return(k)

            q = self.model.encode([query], convert_to_numpy=True)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)  # (1,d)
            n = min(initial, len(self.meta))
            D, I = self.index.search(q.astype(np.float32), n)
            cand_idxs = [int(i) for i in I[0] if i >= 0]

            filtered, texts = [], []
            for idx, score in zip(cand_idxs, D[0].tolist()):
                h = self.meta[idx]
                txt = (h.get("text") or "").strip()
                if not _length_ok(txt) or not _informative(txt):
                    continue
                filtered.append({**h, "score": float(score)})
                texts.append(txt)

            if not filtered:
                return []

            emb = self.model.encode(texts, convert_to_numpy=True)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # (m,d)
            sel = _mmr(q[0], emb, lambda_mult=lambda_mmr, k=min(k, len(filtered)))
            return [filtered[i] for i in sel]

        # SBERT-only path (no FAISS)
        filtered = self._filter_meta()
        if not filtered:
            return []

        # Cap the working set to avoid big encode stalls
        cap = min(max(initial, k), len(filtered), 400)  # 400 keeps it snappy on laptops
        pool = filtered[:cap]

        try:
            self._load_model()
            if not self.model:
                # SBERT unavailable -> naive
                return pool[:k]

            texts = [(h.get("text") or "").strip() for h in pool]
            q = self.model.encode([query], convert_to_numpy=True)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            emb = self.model.encode(texts, convert_to_numpy=True)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            sel = _mmr(q[0], emb, lambda_mult=lambda_mmr, k=min(k, len(pool)))
            return [pool[i] for i in sel]
        except Exception:
            # absolute last resort
            return pool[:k]