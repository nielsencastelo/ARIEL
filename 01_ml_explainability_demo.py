"""
01_ml_explainability_demo.py

Demonstração simples e prática de explicabilidade para ML tradicional.
- treina uma árvore de decisão no Iris
- mostra importância das features
- explica a decisão de uma amostra com o caminho de regras
- salva um JSON com a trilha de decisão

Uso:
    python 01_ml_explainability_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


OUTPUT_DIR = Path("sample_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def explain_sample(model: DecisionTreeClassifier, feature_names: List[str], sample: List[float]) -> Dict[str, Any]:
    tree = model.tree_
    node_indicator = model.decision_path([sample])
    leaf_id = model.apply([sample])[0]

    class_id = int(model.predict([sample])[0])
    proba = model.predict_proba([sample])[0].tolist()

    rules = []
    node_index = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]

    for node_id in node_index:
        if node_id == leaf_id:
            continue

        feature_idx = tree.feature[node_id]
        threshold = float(tree.threshold[node_id])
        feature_name = feature_names[feature_idx]
        sample_value = float(sample[feature_idx])

        go_left = sample_value <= threshold
        operator = "<=" if go_left else ">"
        rules.append(
            {
                "node_id": int(node_id),
                "feature": feature_name,
                "sample_value": round(sample_value, 4),
                "operator": operator,
                "threshold": round(threshold, 4),
                "decision": f"{feature_name} {operator} {threshold:.4f}",
            }
        )

    return {
        "predicted_class_id": class_id,
        "predicted_probabilities": [round(p, 6) for p in proba],
        "rules_path": rules,
    }


def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    sample_index = 0
    sample = X_test[sample_index].tolist()
    explanation = explain_sample(model, feature_names, sample)

    explanation["dataset"] = "iris"
    explanation["feature_names"] = feature_names
    explanation["class_names"] = class_names
    explanation["sample"] = {name: round(float(value), 4) for name, value in zip(feature_names, sample)}
    explanation["accuracy_test"] = round(float(accuracy), 6)
    explanation["predicted_class_name"] = class_names[explanation["predicted_class_id"]]
    explanation["expected_class_name"] = class_names[int(y_test[sample_index])]
    explanation["feature_importances"] = {
        feature_names[i]: round(float(model.feature_importances_[i]), 6)
        for i in range(len(feature_names))
    }

    out_path = OUTPUT_DIR / "ml_explanation.json"
    out_path.write_text(json.dumps(explanation, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("DEMO 1 - EXPLICABILIDADE DE ML")
    print("=" * 80)
    print(f"Acurácia no teste: {accuracy:.4f}")
    print(f"Classe prevista: {explanation['predicted_class_name']}")
    print(f"Classe esperada: {explanation['expected_class_name']}")
    print("\nImportância das features:")
    for k, v in explanation["feature_importances"].items():
        print(f" - {k}: {v}")

    print("\nCaminho de decisão da amostra:")
    for step, rule in enumerate(explanation["rules_path"], start=1):
        print(f" {step}. {rule['decision']} (valor da amostra={rule['sample_value']})")

    print(f"\nJSON salvo em: {out_path.resolve()}")


if __name__ == "__main__":
    main()
