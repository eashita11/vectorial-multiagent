# src/persona_discovery.py
from __future__ import annotations
import os, json, re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

DATA_LINES = Path("data/processed/lines.csv")
OUT_DIR = Path("data/processed/personas")
ANALYSIS_JSON = Path("data/processed/persona_analysis.json")

# --- Regex patterns for tone/style ---
HEDGE_RE = re.compile(r"\b(maybe|perhaps|i think|i guess|it seems|might|could|sort of|kind of|possibly|likely|probably)\b", re.I)
CERTAINTY_RE = re.compile(r"\b(definitely|certainly|must|always|never|clearly|obvious|evident)\b", re.I)
IMPER_RE = re.compile(r"\b(should|need to|have to|do not|don['’]t|please|must)\b", re.I)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]
    df = df[~df["text"].str.match(r"^[\W_]+$")]
    return df

def compute_character_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all lines by character and compute semantic + stylistic features.
    """
    sia = SentimentIntensityAnalyzer()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    features = []
    characters = []
    texts = []

    for char, group in df.groupby("character"):
        lines = group["text"].tolist()
        if len(lines) < 5:
            continue  # skip minor characters
        joined = " ".join(lines[:200])  # cap to avoid memory blow-up
        emb = model.encode(joined, normalize_embeddings=True)
        low = joined.lower()
        sent = sia.polarity_scores(joined)

        # stylistic cues
        question_ratio = joined.count("?") / max(1, len(lines))
        exclaim_ratio = joined.count("!") / max(1, len(lines))
        hedge = len(HEDGE_RE.findall(low)) / max(1, len(lines))
        certainty = len(CERTAINTY_RE.findall(low)) / max(1, len(lines))
        imperative = len(IMPER_RE.findall(low)) / max(1, len(lines))
        word_count = len(joined.split())

        feats = np.concatenate([
            emb,
            np.array([
                question_ratio, exclaim_ratio, hedge, certainty,
                imperative, sent["neg"], sent["neu"], sent["pos"], sent["compound"], word_count/1000
            ])
        ])
        features.append(feats)
        characters.append(char)
        texts.append(joined)

    feats_arr = np.vstack(features)
    dfc = pd.DataFrame({"character": characters, "text": texts})
    return dfc, feats_arr

def cluster_personas(df_char: pd.DataFrame, X: np.ndarray, k=3):
    """
    Cluster characters into k persona groups.
    """
    print(f"🔹 Reducing dimensionality with PCA …")
    pca = PCA(n_components=128, random_state=42)
    Xr = pca.fit_transform(X)

    print(f"🔹 Clustering {len(df_char)} characters into {k} personas …")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xr)
    df_char["cluster"] = labels
    return df_char, labels

def describe_clusters(df_char: pd.DataFrame, k: int):
    summaries = {}
    for i in range(k):
        sub = df_char[df_char["cluster"] == i]
        joined = " ".join(sub["text"].tolist())
        vec = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vec.fit_transform(sub["text"])
        freqs = np.array(X.sum(axis=0)).flatten()
        vocab = np.array(vec.get_feature_names_out())
        top_words = vocab[np.argsort(freqs)[::-1][:15]].tolist()
        summaries[i] = {
            "n_chars": len(sub),
            "example_chars": sub["character"].head(5).tolist(),
            "top_terms": top_words
        }
    return summaries

def discover_personas(k=3):
    assert DATA_LINES.exists(), "Run preprocessing first!"
    df = pd.read_csv(DATA_LINES)
    df = clean_df(df)

    print("🔹 Computing aggregated character embeddings …")
    df_char, X = compute_character_features(df)
    df_char, labels = cluster_personas(df_char, X, k=k)

    summaries = describe_clusters(df_char, k)

    # Save each cluster’s lines
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_map = {0: "commander", 1: "rationalist", 2: "dramatist"} if k == 3 else {i: f"persona_{i}" for i in range(k)}

    for i in range(k):
        chars = df_char[df_char["cluster"] == i]["character"].tolist()
        sub = df[df["character"].isin(chars)]
        out = OUT_DIR / f"{name_map[i]}.csv"
        sub.to_csv(out, index=False)
        print(f"✅ Saved {out} with {len(sub)} lines ({len(chars)} characters)")

    # Write analysis
    analysis = {
        "k": k,
        "clusters": summaries,
        "mapping": name_map
    }
    ANALYSIS_JSON.write_text(json.dumps(analysis, indent=2))
    print(f"📝 Analysis written to {ANALYSIS_JSON}")

if __name__ == "__main__":
    discover_personas(k=3)
