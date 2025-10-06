# tests/test_persona_discovery.py
from pathlib import Path
import pandas as pd

def test_persona_files_exist():
    # Accept either named by role or by cluster index
    candidates = [
        "data/processed/personas/commander.csv",
        "data/processed/personas/rationalist.csv",
        "data/processed/personas/dramatist.csv",
        "data/processed/personas/persona_0.csv",
        "data/processed/personas/persona_1.csv",
        "data/processed/personas/persona_2.csv",
    ]
    assert any(Path(p).exists() for p in candidates), "No persona CSV files found"

def test_persona_file_not_empty():
    # Pick the first existing persona file and check schema
    for p in [
        "data/processed/personas/commander.csv",
        "data/processed/personas/rationalist.csv",
        "data/processed/personas/dramatist.csv",
        "data/processed/personas/persona_0.csv",
        "data/processed/personas/persona_1.csv",
        "data/processed/personas/persona_2.csv",
    ]:
        path = Path(p)
        if path.exists():
            df = pd.read_csv(path)
            assert len(df) > 50
            assert "text" in df.columns
            return
    raise AssertionError("No persona CSV file to validate")