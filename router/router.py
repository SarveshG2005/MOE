"""Classification router for medical and finance queries."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABELS, ROUTER_PATH


class Router:
    def __init__(self):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(ROUTER_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(ROUTER_PATH)
        self.model.to(self.device)
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
            domain: round(float(probabilities[label_id]), 2)
            for label_id, domain in LABELS.items()
        }
        label_id = int(torch.argmax(probabilities).item())
        domain = LABELS[label_id]
        return {
            "domain": domain,
            "confidence": round(float(probabilities[label_id]), 2),
            "scores": scores,
        }
