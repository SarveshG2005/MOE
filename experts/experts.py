"""Causal language model expert used by the MoE system."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MAX_NEW_TOKENS


class Expert:
    MEDICAL_PROMPT = (
        "You are a medical expert. Answer the following medical question "
        "accurately and concisely based on established medical knowledge."
    )
    FINANCE_PROMPT = (
        "You are a financial expert. Answer the following finance question "
        "accurately and concisely based on established financial knowledge."
    )

    def __init__(self, model_name: str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cuda",
        )
        self.model.eval()

    def generate(self, query: str, system_prompt: str) -> str:
        # Merge system prompt into user message — required for Gemma2
        combined = f"{system_prompt}\n\nQuestion: {query}"
        messages = [{"role": "user", "content": combined}]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        generated = output[0, prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()