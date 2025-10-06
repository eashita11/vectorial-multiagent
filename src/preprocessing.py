# src/preprocessing.py
import os
import pandas as pd

def load_movie_lines(path="data/raw/movie_data/movie_lines.tsv"):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # skip header if present
        header = f.readline().strip().split("\t")
        has_header = header[:5] == ["lineID", "characterID", "movieID", "character", "text"]
        if not has_header:
            # first line is data; process it and continue
            parts = header
            # Ensure exactly 5 fields: lineID, characterID, movieID, character, text
            if len(parts) >= 5:
                first = parts[:4]
                text = "\t".join(parts[4:])  # rejoin any extra tabs into text
                rows.append({
                    "line_id": first[0],
                    "char_id": first[1],
                    "movie_id": first[2],
                    "character": first[3],
                    "text": text
                })
        # process rest of file
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                # skip weird short lines
                continue
            first = parts[:4]
            text = "\t".join(parts[4:])  # glue back extra tabs into text field
            rows.append({
                "line_id": first[0],
                "char_id": first[1],
                "movie_id": first[2],
                "character": first[3],
                "text": text
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = load_movie_lines()
    # Basic cleanup
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df.to_csv("data/processed/lines.csv", index=False)
    print("✅ Saved processed lines to data/processed/lines.csv")
    print(df.head())
    print(f"Rows: {len(df):,}")
