# app/agents/tax_calculator.py
"""
Tax Calculator Agent
- exchange_rate_loader / final_cost_calculator 도구를 직접 호출하여 비용 계산
- parallel_fetch에서 이미 환율을 조회한 경우 재조회 없이 재사용
"""
from typing import Dict, Any, Optional
from app.tools import exchange_rate_loader, final_cost_calculator


class TaxCalculatorAgent:
    """환율 조회 → 비용 계산을 순차적으로 수행하는 에이전트."""

    async def run(
        self,
        unit_price: float,
        quantity: int,
        currency: str,
        tariff_rate: float,
        total_foreign_price: Optional[float] = None,
        quantity_unit: str = "개",
        price_unit: str = "1개당",
        exchange_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        환율 조회 및 비용 계산 실행.

        Args:
            unit_price: 물품 단가 (외화)
            quantity: 수량
            currency: 통화 코드 (USD, EUR 등)
            tariff_rate: 관세율 (%)
            total_foreign_price: 총 외화 금액 (단위 변환 후, 있으면 우선 사용)
            quantity_unit: 수량 단위
            price_unit: 단가 기준
            exchange_rate: 이미 조회된 환율 (있으면 재조회 생략)

        Returns:
            {"exchange_rate", "tax_amount", "vat_amount", "total_cost", "breakdown"}
        """
        actual_foreign_total = total_foreign_price if total_foreign_price else (quantity * unit_price)

        print(
            f"[TaxCalculatorAgent] 실행: {quantity}{quantity_unit} × {unit_price} {currency}"
            f" ({price_unit}), 총외화={actual_foreign_total} {currency}, 관세율={tariff_rate}%"
        )

        # 환율 조회 (이미 알고 있으면 스킵)
        if exchange_rate is None:
            rate_result = exchange_rate_loader.invoke({"target_currency": currency})
            exchange_rate = rate_result["rate"]

        # 비용 계산 (total_foreign_price를 수량/단가로 표현하기 위해 quantity=1로 전달)
        calc_result = final_cost_calculator.invoke({
            "item_price": actual_foreign_total,
            "quantity": 1,
            "exchange_rate": exchange_rate,
            "tariff_rate": tariff_rate,
        })

        tax_amount = calc_result["tariff_amount"]
        vat_amount = calc_result["vat_amount"]
        total_cost = calc_result["total_cost"]
        total_krw = actual_foreign_total * exchange_rate

        breakdown = (
            f"--- 최종 비용 계산 결과 ---\n"
            f"1. 총 물품 가격 (외화): {actual_foreign_total:,.2f} {currency}\n"
            f"   ({quantity}{quantity_unit} × {unit_price} {currency} {price_unit})\n"
            f"2. 총 물품 가격 (원화): {total_krw:,.0f} 원\n"
            f"   (총 외화 {actual_foreign_total:,.2f} × 환율 {exchange_rate:,.2f})\n"
            f"3. 예상 관세 ({tariff_rate}%): {tax_amount:,.0f} 원\n"
            f"4. 예상 부가세 (10%): {vat_amount:,.0f} 원\n"
            f"--------------------------------\n"
            f"   총 예상 수입 비용: {total_cost:,.0f} 원\n"
            f"--------------------------------"
        )

        print(f"[TaxCalculatorAgent] 완료: 총 {total_cost:,.0f}원")
        return {
            "exchange_rate": exchange_rate,
            "currency": currency,
            "tax_amount": tax_amount,
            "vat_amount": vat_amount,
            "total_cost": total_cost,
            "breakdown": breakdown,
        }
