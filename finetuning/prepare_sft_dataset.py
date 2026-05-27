import json
from pathlib import Path

src = Path(__file__).resolve().parents[1] / "data" / "qlora_sft_claimmate.jsonl"

def main():
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} SFT examples.")
    print("First example:")
    print(json.dumps(rows[0], indent=2))

if __name__ == "__main__":
    main()
