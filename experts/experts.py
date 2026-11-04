"""Causal language model expert used by the MoE system."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()

    def generate(self, query: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            model_inputs = {"input_ids": input_ids}
        else:
            prompt = f"{system_prompt}\n\nQuestion: {query}\nAnswer:"
            model_inputs = self.tokenizer(prompt, return_tensors="pt")

        model_inputs = {
            name: value.to(self.model.device)
            for name, value in model_inputs.items()
        }
        prompt_length = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        generated = output[0, prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
