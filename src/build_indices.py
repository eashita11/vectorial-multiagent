# src/build_indices.py
import os, json
from pathlib import Path
import numpy as np, faiss, pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PERSONA_DIR = Path("data/processed/personas")

def build_one(csv_path: Path, model):
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    # embed
    embs = []
    for i in tqdm(range(0, len(texts), 128)):
        chunk = texts[i:i+128]
        v = model.encode(chunk, normalize_embeddings=True)
        embs.append(v)
    X = np.vstack(embs).astype("float32")

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    name = csv_path.stem  # persona name from filename
    out_dir = PERSONA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / f"{name}.faiss"))
    # save metadata for citations
    meta = df[["line_id","movie_id","character","text"]].to_dict(orient="records")
    (out_dir / f"{name}.meta.jsonl").write_text("\n".join(json.dumps(x) for x in meta))
    print(f"✅ Built {name}: {len(df):,} vectors to {out_dir}/")

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    csvs = sorted(PERSONA_DIR.glob("*.csv"))
    if not csvs:
        raise SystemExit("No persona CSVs found in data/processed/personas/*.csv")
    for p in csvs:
        build_one(p, model)

if __name__ == "__main__":
    main()
