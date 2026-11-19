"""Classification router for medical and finance queries."""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABELS, ROUTER_PATH


class Router:
    def __init__(self):
        # Resolve path — works locally and on Kaggle
        if os.path.exists(ROUTER_PATH):
            path = ROUTER_PATH
        elif os.path.exists("/kaggle/input/datasets/sarveshg2005/moe-router-final"):
            path = "/kaggle/input/datasets/sarveshg2005/moe-router-final"
        else:
            raise FileNotFoundError(f"Router not found at {ROUTER_PATH}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=2, local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, query: str) -> dict[str, object]:
        inputs = self.tokenizer(
            query,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        scores = {
            domain: round(float(probabilities[label_id]), 4)
            for label_id, domain in LABELS.items()
        }
        label_id = int(torch.argmax(probabilities).item())
        domain = LABELS[label_id]
        return {
            "domain": domain,
            "confidence": round(float(probabilities[label_id]), 4),
            "scores": scores,
        }