"""Fine-tune a DeBERTa router from data/router_data.csv."""

import sys
from pathlib import Path

import numpy as np
from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, ROUTER_PATH


def main() -> None:
    # ── Load dataset ──────────────────────────────────────────────────────
    csv_path = Path("data/router_data.csv")
    dataset = Dataset.from_csv(str(csv_path)).cast_column(
        "label", ClassLabel(num_classes=len(LABELS))
    )

    # ── Stratified 80/20 split ────────────────────────────────────────────
    train_idx, eval_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        stratify=dataset["label"],
        random_state=42,
    )
    train_data = dataset.select(train_idx)
    eval_data  = dataset.select(eval_idx)

    # ── Tokenise ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def tokenize(batch):
        return tokenizer(batch["query"], truncation=True, max_length=512)

    train_data = train_data.map(tokenize, batched=True)
    eval_data  = eval_data.map(tokenize, batched=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=2,
        id2label=LABELS,
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    def metrics(eval_prediction):
        predictions = np.argmax(eval_prediction.predictions, axis=-1)
        acc = accuracy_score(eval_prediction.label_ids, predictions)
        f1  = f1_score(eval_prediction.label_ids, predictions, average="macro")
        return {"accuracy": acc, "f1": f1}

    # ── Training arguments ────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir                  = "router/checkpoints",
        num_train_epochs            = 10,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 16,
        learning_rate               = 2e-5,
        warmup_ratio                = 0.1,
        fp16                        = True,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        save_total_limit            = 1,
        report_to                   = "none",
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_data,
        eval_dataset    = eval_data,
        tokenizer       = tokenizer,
        data_collator   = DataCollatorWithPadding(tokenizer),
        compute_metrics = metrics,
    )

    trainer.train()

    # ── Save best model ───────────────────────────────────────────────────
    trainer.save_model(ROUTER_PATH)
    tokenizer.save_pretrained(ROUTER_PATH)

    # ── Final evaluation ──────────────────────────────────────────────────
    predictions      = trainer.predict(eval_data)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)

    print(f"\nFinal Accuracy : {accuracy_score(predictions.label_ids, predicted_labels):.4f}")
    print(f"Final F1 Macro : {f1_score(predictions.label_ids, predicted_labels, average='macro'):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            predictions.label_ids,
            predicted_labels,
            labels       = list(LABELS),
            target_names = list(LABELS.values()),
            zero_division = 0,
        )
    )


if __name__ == "__main__":
    main()