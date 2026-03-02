"""
eval/dataset/generate_cases.py
tariff_by_hs.csv에서 추가 테스트 케이스를 자동 생성

  python eval/dataset/generate_cases.py --n 20 --output eval/dataset/auto_cases.json

생성 전략
─────────
  - 류(chapter) 별로 골고루 샘플
  - 한글품명이 명확한 세부 세번(9-10자리) 행 우선
  - 중복 류는 제외
"""

import sys
import json
import random
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def normalize_hs(hs_code: str, digits: int = 6) -> str:
    cleaned = re.sub(r"[.\-\s]", "", str(hs_code))
    return cleaned.zfill(10)[:digits]


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description="CSV로부터 테스트 케이스 자동 생성")
    parser.add_argument("--n", type=int, default=20, help="생성할 케이스 수 (기본: 20)")
    parser.add_argument(
        "--output",
        default=str(ROOT / "eval" / "dataset" / "auto_cases.json"),
        help="저장 경로",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    csv_path = ROOT / "data" / "tariff_by_hs.csv"
    if not csv_path.exists():
        print(f"CSV 파일 없음: {csv_path}")
        return

    df = pd.read_csv(str(csv_path), encoding="cp949", dtype={"세번": str})

    # 세부 세번 (9-10자리)만, 한글품명 있는 행
    df = df.dropna(subset=["세번", "한글품명"])
    df["세번_len"] = df["세번"].astype(str).str.len()
    df = df[df["세번_len"] >= 9]

    # 류 추출
    df["chapter"] = df["세번"].astype(str).str.zfill(10).str[:2]

    # 류별 1개씩 골고루 샘플
    random.seed(args.seed)
    chapters = sorted(df["chapter"].unique())
    random.shuffle(chapters)

    cases = []
    used_chapters: set = set()

    for ch in chapters:
        if len(cases) >= args.n:
            break
        ch_df = df[df["chapter"] == ch].sample(frac=1, random_state=args.seed)
        row = ch_df.iloc[0]

        item_name = str(row.get("한글품명", "")).strip()
        if not item_name or item_name in ("nan", ""):
            continue

        hs6 = normalize_hs(str(row["세번"]), 6)

        cases.append(
            {
                "id":              len(cases) + 1,
                "item_name":       item_name,
                "description":     f"{item_name} 수입",
                "expected_hs6":    hs6,
                "expected_chapter": ch,
                "category":        "auto",
                "difficulty":      2,
                "source_세번":     str(row["세번"]),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"{len(cases)}개 케이스 생성 → {out}")
    for c in cases[:5]:
        print(f"  [{c['id']:02d}] {c['item_name']:<20} HS6={c['expected_hs6']}")
    if len(cases) > 5:
        print(f"  ... (이하 {len(cases)-5}개)")


if __name__ == "__main__":
    main()
