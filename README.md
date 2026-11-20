---
title: Medical Finance Moe
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Medical · Finance Mixture of Experts

## Overview

This project is a Mixture-of-Experts system that classifies each user query
with a fine-tuned DeBERTa router and sends it to a specialized medical or
finance language model. It was built to demonstrate accurate domain routing,
specialized answers, and transparent expert-load monitoring when terminology
overlaps across professional fields.

## Architecture

```text
Query → Router (DeBERTa-v3-base, 86M) → Medical Expert (Meditron3-Gemma2-2B)
                                      → Finance Expert (Qwen2.5-3B-Instruct)
```

The router returns a domain, confidence, and both domain scores. The selected
expert then generates the final response while the system tracks how many
queries each expert has handled.

## Why Medical + Finance

Medical and finance questions often share terminology, making keyword-only
routing unreliable. Words such as exposure, yield, margin, discharge, stress,
bond, and premium can describe either a clinical concept or a financial
concept depending on the surrounding query. A trained router is therefore
needed to use context rather than simply matching domain-specific words.

## Hardware Requirements

| Component | Size | Precision | VRAM |
|---|---:|---|---:|
| Router | 86M | FP16 | 0.17 GB |
| Medical | 2B | FP16 | 4.0 GB |
| Finance | 3B | FP16 | 6.0 GB |
| Overhead |  |  | 1.5 GB |
| Total |  |  | ~11.7 GB |

A T4 with 16 GB of VRAM fits this configuration with approximately 4.3 GB of
headroom.

## Validation

- 80/20 stratified train/validation split
- Validation every epoch
- Best checkpoint selected on F1 macro
- Separate held-out benchmark: 40 queries (15 medical, 15 finance, 10 ambiguous)

## Metrics

- Router: Accuracy, Precision, Recall, F1 per class and macro, Confusion matrix
- Experts: Response latency (seconds/query), BLEU score

## Results

```text
Router Accuracy : TBD
Router F1 Macro : TBD
Medical latency : TBD
Finance latency : TBD
```

## How to Run

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train and save the router:

   ```bash
   python router/train_router.py
   ```

3. Run the held-out benchmark:

   ```bash
   python benchmark.py
   ```

   This writes `results/accuracy.json` and `results/load_balance.png`.

4. Launch the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Models

| Model name | Size | Domain | HuggingFace |
|---|---:|---|---|
| DeBERTa-v3-base | 86M | Router | [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) |
| Meditron3-Gemma2-2B | 2B | Medical | [EPFLiGHT/Meditron3-Gemma2-2B](https://huggingface.co/EPFLiGHT/Meditron3-Gemma2-2B) |
| Qwen2.5-3B-Instruct | 3B | Finance | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |
