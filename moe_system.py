"""Mixture-of-Experts orchestration and load-balancing statistics."""

from config import FINANCE_MODEL, MEDICAL_MODEL
from experts.experts import Expert
from router.router import Router


class MoESystem:
    def __init__(self):
        self.router = Router()
        self.medical_expert = Expert(MEDICAL_MODEL)
        self.finance_expert = Expert(FINANCE_MODEL)
        self.load_counts = {"medical": 0, "finance": 0}

    def query(self, user_input: str) -> dict:
        prediction = self.router.predict(user_input)
        domain = prediction["domain"]
        confidence = prediction["confidence"]
        scores = prediction["scores"]
        self.load_counts[domain] += 1
        if domain == "medical":
            expert = self.medical_expert
            system_prompt = Expert.MEDICAL_PROMPT
        else:
            expert = self.finance_expert
            system_prompt = Expert.FINANCE_PROMPT

        response = expert.generate(user_input, system_prompt)
        print(f"[Router] \u2192 {domain} (confidence: {confidence:.2%})")
        return {
            "domain": domain,
            "confidence": confidence,
            "scores": scores,
            "response": response,
            "load": self.load_counts.copy(),
        }
