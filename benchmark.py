"""Benchmark router accuracy and save load-balancing visualizations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import LABELS
from router.router import Router


def main() -> None:
    data = pd.read_csv(Path("data") / "benchmark.csv")
    router = Router()
    expected = data["expected"].tolist()
    predicted = [
        router.predict(query)["domain"]
        for query in data["query"].astype(str)
    ]
    labels = [LABELS[label_id] for label_id in sorted(LABELS)]
    accuracy = accuracy_score(expected, predicted)
    f1_macro = f1_score(
        expected, predicted, labels=labels, average="macro", zero_division=0
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(labels)
    }
    matrix = confusion_matrix(expected, predicted, labels=labels)

    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"F1 macro score: {f1_macro:.4f}")
    for label in labels:
        print(
            f"{label} - precision: {per_class[label]['precision']:.4f}, "
            f"recall: {per_class[label]['recall']:.4f}, "
            f"F1: {per_class[label]['f1']:.4f}"
        )
    print("Confusion matrix:")
    print("               " + "  ".join(f"{label:>8}" for label in labels))
    for label, row in zip(labels, matrix):
        print(f"{label:>8}       " + "  ".join(f"{value:>8}" for value in row))

    output = Path("results")
    output.mkdir(exist_ok=True)
    (output / "accuracy.json").write_text(
        json.dumps(
            {
                "accuracy": float(accuracy),
                "f1_macro": float(f1_macro),
                "medical": per_class["medical"],
                "finance": per_class["finance"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    counts = pd.Series(predicted).value_counts().reindex(labels, fill_value=0)
    figure, axis = plt.subplots()
    axis.bar(labels, counts.values)
    axis.set_xlabel("Domain")
    axis.set_ylabel("Number of queries routed")
    axis.set_title("Expert Load Distribution \u2014 Benchmark Run")
    figure.tight_layout()
    figure.savefig(output / "load_balance.png")
    plt.close(figure)


if __name__ == "__main__":
    main()
