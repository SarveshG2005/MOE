"""Central configuration for the Medical-Finance MoE system."""

# ── Router ────────────────────────────────────────────────────────────────────
LABELS   = {0: "medical", 1: "finance"}   # id → domain name
LABEL2ID = {"medical": 0, "finance": 1}   # domain name → id
ROUTER_PATH = "router/final"              # saved router directory

# ── Expert models (swap here if a model is unavailable) ──────────────────────
MEDICAL_MODEL = "EPFLiGHT/Meditron3-Gemma2-2B"
FINANCE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 200

# ── Hardware reference (T4 = 16 GB VRAM) ─────────────────────────────────────
# Router  (86M  FP16): 0.17 GB
# Medical (2B   FP16): 4.0  GB
# Finance (3B   FP16): 6.0  GB
# Overhead           : 1.5  GB
# Total peak         : ~11.7 GB  →  fits T4 with 4.3 GB headroom