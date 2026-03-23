"""
02_policy_reasoning_demo.py

Demonstração de raciocínio híbrido:
- extrai fatos simples de um texto
- aplica regras simbólicas
- produz uma explicação auditável

Uso:
    python 02_policy_reasoning_demo.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


OUTPUT_DIR = Path("sample_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class CaseFacts:
    age: int
    urgent: bool
    consent: bool
    amount: float


def parse_case_text(text: str) -> CaseFacts:
    age_match = re.search(r"(\d+)\s*anos", text, flags=re.IGNORECASE)
    amount_match = re.search(r"R\$\s*([0-9]+(?:[.,][0-9]+)?)", text, flags=re.IGNORECASE)

    age = int(age_match.group(1)) if age_match else 0
    raw_amount = amount_match.group(1).replace(".", "").replace(",", ".") if amount_match else "0"
    amount = float(raw_amount)

    urgent = bool(re.search(r"urgente|urgência", text, flags=re.IGNORECASE))
    consent = bool(re.search(r"consentimento|autorização assinada", text, flags=re.IGNORECASE))

    return CaseFacts(age=age, urgent=urgent, consent=consent, amount=amount)


def evaluate_policy(facts: CaseFacts) -> Dict[str, object]:
    trace: List[Dict[str, object]] = []

    def add(rule_id: str, description: str, passed: bool) -> None:
        trace.append({
            "rule_id": rule_id,
            "description": description,
            "passed": passed,
        })

    add("R1", "Valor máximo permitido sem comitê é R$ 5.000,00", facts.amount <= 5000)
    add("R2", "Casos urgentes podem seguir fluxo acelerado", facts.urgent)
    add("R3", "Menor de idade exige consentimento", (facts.age >= 18) or facts.consent)

    approved = all(item["passed"] for item in trace)
    blockers = [item["description"] for item in trace if not item["passed"]]

    if approved:
        decision = "APPROVED"
        reason = "Todas as regras obrigatórias foram satisfeitas."
    else:
        decision = "REVIEW_REQUIRED"
        reason = "Há restrições não satisfeitas que exigem revisão humana."

    return {
        "decision": decision,
        "reason": reason,
        "facts": asdict(facts),
        "trace": trace,
        "blockers": blockers,
    }


def main() -> None:
    sample_text = (
        "Paciente de 16 anos, caso urgente, valor solicitado de R$ 4200, "
        "com consentimento assinado pelo responsável."
    )

    facts = parse_case_text(sample_text)
    result = evaluate_policy(facts)

    out_path = OUTPUT_DIR / "policy_reasoning.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("DEMO 2 - RACIOCÍNIO HÍBRIDO COM REGRAS")
    print("=" * 80)
    print(f"Texto de entrada: {sample_text}")
    print("\nFatos extraídos:")
    for k, v in result["facts"].items():
        print(f" - {k}: {v}")

    print("\nRegras avaliadas:")
    for row in result["trace"]:
        status = "OK" if row["passed"] else "FALHOU"
        print(f" - [{status}] {row['rule_id']}: {row['description']}")

    print(f"\nDecisão final: {result['decision']}")
    print(f"Justificativa: {result['reason']}")
    print(f"JSON salvo em: {out_path.resolve()}")


if __name__ == "__main__":
    main()
