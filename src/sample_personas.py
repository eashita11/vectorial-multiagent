import pandas as pd

for i in [0, 1, 2]:
    df = pd.read_csv(f"data/processed/personas/persona_{i}.csv")
    print(f"\n--- persona_{i} sample ---")
    print("\n".join(df["text"].dropna().sample(10, random_state=42).tolist()))
