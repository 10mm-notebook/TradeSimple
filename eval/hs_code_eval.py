import asyncio
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

from app.agents.hs_code_finder import HSCodeFinderAgent


@dataclass
class EvalCase:
    id: int
    item_name: str
    description: str
    expected_hs: str


def load_eval_cases(path: str) -> List[EvalCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases: List[EvalCase] = []
    for row in raw:
        cases.append(
            EvalCase(
                id=row["id"],
                item_name=row["item_name"],
                description=row.get("description", row["item_name"]),
                expected_hs=row["expected_hs"],
            )
        )
    return cases


def normalize_hs(hs: str) -> str:
    """HS 코드를 비교용으로 정규화 (숫자만 남김)."""
    return "".join(ch for ch in hs if ch.isdigit())


async def eval_single_case(agent: HSCodeFinderAgent, case: EvalCase) -> Dict[str, Any]:
    """단일 테스트 케이스 평가."""
    # 포트폴리오용: description을 그대로 넣되, 실제 프롬프트 설계는 에이전트 내부에서 수행
    result = await agent.run(case.description)
    predicted_hs = result.get("hs_code", "") or ""

    expected_norm = normalize_hs(case.expected_hs)
    predicted_norm = normalize_hs(predicted_hs)

    exact_10 = expected_norm == predicted_norm and len(expected_norm) == 10
    # 6단위(앞 6자리) 부분 일치
    partial_6 = (
        len(expected_norm) >= 6
        and len(predicted_norm) >= 6
        and expected_norm[:6] == predicted_norm[:6]
    )

    return {
        "id": case.id,
        "item_name": case.item_name,
        "description": case.description,
        "expected_hs": case.expected_hs,
        "predicted_hs": predicted_hs,
        "exact_10_match": exact_10,
        "partial_6_match": partial_6,
        "raw_rationale": result.get("rationale", ""),
    }


async def main() -> None:
    """
    HS 코드 분류 에이전트 정확도 평가 스크립트.

    - data/hs_eval_cases.json 에 정의된 테스트 케이스를 순회
    - 각 케이스에 대해 HSCodeFinderAgent.run() 호출
    - 10단위 완전 일치 / 6단위 부분 일치 정확도를 계산
    - 결과를 eval/results_hs_code.json 에 저장
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    cases_path = os.path.join(base_dir, "data", "hs_eval_cases.json")
    output_path = os.path.join(base_dir, "eval", "results_hs_code.json")

    cases = load_eval_cases(cases_path)
    agent = HSCodeFinderAgent()

    results: List[Dict[str, Any]] = []

    for case in cases:
        print(f"[EVAL] Case {case.id}: {case.item_name}")
        res = await eval_single_case(agent, case)
        results.append(res)
        # OpenAI rate limit 완화를 위한 짧은 대기 (필요 시 조정)
        await asyncio.sleep(0.5)

    # 메트릭 계산
    total = len(results)
    exact_10 = sum(1 for r in results if r["exact_10_match"])
    partial_6 = sum(1 for r in results if r["partial_6_match"])

    metrics = {
        "total_cases": total,
        "exact_10_matches": exact_10,
        "partial_6_matches": partial_6,
        "exact_10_accuracy": exact_10 / total if total else 0.0,
        "partial_6_accuracy": partial_6 / total if total else 0.0,
    }

    summary = {
        "metrics": metrics,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[EVAL] HS 코드 분류 평가 완료")
    print(f"- 총 케이스 수: {total}")
    print(f"- 10단위 정확 일치: {exact_10} / {total} ({metrics['exact_10_accuracy']*100:.1f}%)")
    print(f"- 6단위 부분 일치: {partial_6} / {total} ({metrics['partial_6_accuracy']*100:.1f}%)")
    print(f"- 상세 결과: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

