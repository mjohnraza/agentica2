"""Self-Hosting Memory, KV-Cache Architecture, and GPU Concurrency Analysis.

Section 2, Item 4 & Item 5 of SaafHawa Agent Assignment.
Model evaluated: Meta Llama 3.1 8B Instruct (Grouped-Query Attention).
Reads architecture configuration directly from Hugging Face config.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from tabulate import tabulate

CONFIG_PATH = Path("config.json")

# Student Derived Parameters (23K-0069 -> R = 69, D = 0)
Q_NORMAL = 3225       # Normal daily queries
N_CALLS = 6           # LLM calls per query
TOOL_P = 0.04         # Failure probability
H_DAYS = 5            # Heatwave days per month
BUDGET_USD = 1890.0   # Monthly budget
TARGET_T = 10.0       # Target latency (seconds)

# Longest Context in Agent Loop:
# Call k=1..N (N=6): input = 2000 + 800*(k-1)
# For k=6: input = 2000 + 800*5 = 6000 tokens. Output = 600 tokens.
# Longest sequence length = 6000 + 600 = 6,600 tokens.
LONGEST_SEQUENCE_LENGTH = 6600


def load_model_architecture():
    """Loads and derives architectural parameters from config.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found in working directory.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg.get("_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")
    num_layers = cfg.get("num_hidden_layers", 32)
    num_heads = cfg.get("num_attention_heads", 32)
    num_kv_heads = cfg.get("num_key_value_heads", 8)
    hidden_size = cfg.get("hidden_size", 4096)

    if "head_dim" in cfg:
        head_dim = cfg["head_dim"]
        head_dim_source = "explicit in config.json"
    else:
        head_dim = hidden_size // num_heads
        head_dim_source = f"derived (hidden_size // num_attention_heads = {hidden_size} // {num_heads})"

    total_params = 8.03e9  # 8.03 Billion parameters for Llama 3.1 8B

    return {
        "model_name": model_name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "head_dim_source": head_dim_source,
        "total_params": total_params,
        "attention_type": "Grouped-Query Attention (GQA)",
        "gqa_ratio": f"{num_heads // num_kv_heads}:1 query-to-KV head sharing",
    }


def calculate_weight_memory(total_params: float):
    """Calculates model weight memory across precisions."""
    mem_fp16_bytes = total_params * 2.0
    mem_int8_bytes = total_params * 1.0
    # 4-bit weights with ~5% quantization scale & zero-point metadata overhead
    mem_int4_bytes = total_params * 0.5 * 1.05

    return {
        "fp16_gb": round(mem_fp16_bytes / (1024**3), 2),
        "int8_gb": round(mem_int8_bytes / (1024**3), 2),
        "int4_gb": round(mem_int4_bytes / (1024**3), 2),
    }


def calculate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    sequence_length: int = LONGEST_SEQUENCE_LENGTH,
    bytes_per_value: int = 2,  # FP16 KV cache
):
    """Calculates KV cache memory per sequence using assignment formula.

    KV bytes = 2 * layers * kv_heads * head_dim * sequence_length * bytes_per_value
    """
    kv_bytes = 2 * num_layers * num_kv_heads * head_dim * sequence_length * bytes_per_value
    kv_mib = kv_bytes / (1024**2)
    kv_gb = kv_bytes / (1024**3)

    return {
        "sequence_length": sequence_length,
        "bytes_per_value": bytes_per_value,
        "kv_bytes": kv_bytes,
        "kv_mib": round(kv_mib, 2),
        "kv_gb": round(kv_gb, 4),
    }


def calculate_gpu_concurrency(
    gpu_name: str,
    total_vram_gb: float,
    weight_int4_gb: float,
    kv_cache_per_seq_gb: float,
    overhead_pct: float = 0.10,
):
    """Calculates maximum concurrent sequences on a real GPU."""
    overhead_gb = total_vram_gb * overhead_pct
    usable_vram_gb = total_vram_gb - overhead_gb
    remaining_kv_vram_gb = usable_vram_gb - weight_int4_gb

    if remaining_kv_vram_gb <= 0:
        max_concurrent = 0
    else:
        max_concurrent = int(remaining_kv_vram_gb // kv_cache_per_seq_gb)

    return {
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "overhead_gb": round(overhead_gb, 2),
        "usable_vram_gb": round(usable_vram_gb, 2),
        "weight_int4_gb": weight_int4_gb,
        "remaining_kv_vram_gb": round(remaining_kv_vram_gb, 2),
        "max_concurrent_sequences": max_concurrent,
    }


def calculate_peak_sessions_and_gpus(
    max_concurrent_per_gpu: int,
    agent_latency_sec: float = 7.85,
):
    """Calculates active peak concurrent queries using Little's Law."""
    # Peak query arrival rate during heatwave window (12:00 - 14:00):
    # Daily heatwave queries = 5 * 3,225 = 16,125
    # 35% in 2 hours = 5,643.75 queries / 120 mins = 47.03 queries/minute
    peak_qpm = 47.03125
    arrival_rate_per_sec = peak_qpm / 60.0  # ~0.7838 queries/sec

    # Little's Law: L = lambda * W
    # L: Average concurrent queries in flight
    active_concurrent_sessions = arrival_rate_per_sec * agent_latency_sec
    peak_active_sessions_ceiling = math.ceil(active_concurrent_sessions)

    # Number of GPUs required
    gpus_required = math.ceil(peak_active_sessions_ceiling / max_concurrent_per_gpu)
    # High availability (N + 1) redundancy recommendation
    gpus_recommended_ha = gpus_required + 1

    return {
        "peak_qpm": round(peak_qpm, 2),
        "arrival_rate_per_sec": round(arrival_rate_per_sec, 4),
        "agent_latency_sec": agent_latency_sec,
        "active_concurrent_sessions": round(active_concurrent_sessions, 2),
        "peak_active_sessions_ceiling": peak_active_sessions_ceiling,
        "max_concurrent_per_gpu": max_concurrent_per_gpu,
        "gpus_required": gpus_required,
        "gpus_recommended_ha": gpus_recommended_ha,
    }


def run_break_even_analysis(
    gpu_monthly_cost_usd: float = 248.20,  # e.g. RunPod / Lambda RTX 4090 @ $0.34/hr * 730 hr
    gpu_capacity_concurrent: int = 20,
    cloud_api_cost_per_query: float = 0.004575,  # GPT-4o-mini with caching ($737.72 / 161,250 queries)
):
    """Section 2, Item 5: Break-even analysis modeling GPU costs as a step function."""
    # Step function: As daily query volume Q grows, calculate required GPUs and total hosting cost
    # Compare with API cost across volumes
    query_volumes = [500, 1000, 1500, 2000, 3225, 5000, 7500, 10000, 15000]
    comparison_rows = []

    for q in query_volumes:
        # Monthly queries (with 5 heatwave days at 5x volume)
        m_queries = (25 * q) + (5 * 5 * q)
        # Peak QPM
        p_qpm = (0.35 * 5 * q) / 120.0
        # Active sessions (Little's Law @ 7.85s latency)
        active_sess = math.ceil((p_qpm / 60.0) * 7.85)
        # GPUs needed (step function)
        gpus = max(1, math.ceil(active_sess / gpu_capacity_concurrent))
        self_host_cost = gpus * gpu_monthly_cost_usd
        api_cost = m_queries * cloud_api_cost_per_query
        diff = api_cost - self_host_cost

        comparison_rows.append({
            "daily_q": q,
            "monthly_queries": m_queries,
            "peak_qpm": round(p_qpm, 2),
            "active_sessions": active_sess,
            "gpus_needed": gpus,
            "self_host_cost": round(self_host_cost, 2),
            "api_cost": round(api_cost, 2),
            "cheaper": "Self-Hosting" if diff > 0 else "Cloud API",
            "net_savings": round(abs(diff), 2),
        })

    # Exact crossover point Q* for 1 GPU
    # 1 GPU cost = $248.20/month
    # Total monthly queries = 50 * Q_daily
    # 50 * Q_daily * cloud_api_cost_per_query = gpu_monthly_cost_usd
    # Q* = gpu_monthly_cost_usd / (50 * cloud_api_cost_per_query)
    q_star_daily = gpu_monthly_cost_usd / (50.0 * cloud_api_cost_per_query)

    return {
        "gpu_monthly_cost_usd": gpu_monthly_cost_usd,
        "cloud_api_cost_per_query": cloud_api_cost_per_query,
        "q_star_daily": round(q_star_daily, 0),
        "comparison_table": comparison_rows,
    }


def main():
    print("=" * 80)
    print("  SaafHawa Agent - Section 2, Item 4 & 5: Self-Hosting Memory & KV Cache")
    print("=" * 80)

    # 1. Architecture Parameters
    arch = load_model_architecture()
    print("\n--- 1. MODEL ARCHITECTURE FROM config.json ---")
    print(f"• Model:                 {arch['model_name']}")
    print(f"• Total Parameters:      {arch['total_params'] / 1e9:.2f} Billion")
    print(f"• Hidden Layers:         {arch['num_layers']}")
    print(f"• Attention Heads:       {arch['num_heads']}")
    print(f"• Key-Value Heads:       {arch['num_kv_heads']}")
    print(f"• Hidden Size:           {arch['hidden_size']}")
    print(f"• Head Dimension:        {arch['head_dim']} ({arch['head_dim_source']})")
    print(f"• Attention Paradigm:    {arch['attention_type']} ({arch['gqa_ratio']})")

    # 2. Weight Memory
    w_mem = calculate_weight_memory(arch["total_params"])
    print("\n--- 2. WEIGHT MEMORY REQUIREMENTS ---")
    print(f"• FP16 (16-bit, 2 bytes/param):   {w_mem['fp16_gb']} GB")
    print(f"• INT8 (8-bit, 1 byte/param):     {w_mem['int8_gb']} GB")
    print(f"• INT4 (4-bit, 0.5 byte + meta):  {w_mem['int4_gb']} GB")

    # 3. KV-Cache Memory per Sequence
    kv_mem = calculate_kv_cache_memory(
        num_layers=arch["num_layers"],
        num_kv_heads=arch["num_kv_heads"],
        head_dim=arch["head_dim"],
        sequence_length=LONGEST_SEQUENCE_LENGTH,
    )
    print("\n--- 3. KV-CACHE MEMORY (Longest Context = 6,600 Tokens) ---")
    print(f"• KV Cache Formula:       2 * layers * kv_heads * head_dim * seq_len * 2 bytes")
    print(f"• Calculation:            2 * {arch['num_layers']} * {arch['num_kv_heads']} * {arch['head_dim']} * {LONGEST_SEQUENCE_LENGTH} * 2")
    print(f"• KV Cache per Sequence:  {kv_mem['kv_bytes']:,} bytes ({kv_mem['kv_mib']} MiB / {kv_mem['kv_gb']} GB)")

    # 4. GPU Concurrency
    gpus_to_eval = [
        {"name": "NVIDIA RTX 4090 / A10G", "vram": 24.0},
        {"name": "NVIDIA A100 (SXM4)", "vram": 80.0},
    ]
    print("\n--- 4. GPU CONCURRENCY & CAPACITY (4-bit Weights + FP16 KV Cache, 10% Reserve) ---")
    gpu_results = []
    for g in gpus_to_eval:
        conc = calculate_gpu_concurrency(
            gpu_name=g["name"],
            total_vram_gb=g["vram"],
            weight_int4_gb=w_mem["int4_gb"],
            kv_cache_per_seq_gb=kv_mem["kv_gb"],
        )
        gpu_results.append(conc)

    print(tabulate(
        [
            [
                r["gpu_name"],
                f"{r['total_vram_gb']} GB",
                f"{r['overhead_gb']} GB",
                f"{r['weight_int4_gb']} GB",
                f"{r['remaining_kv_vram_gb']} GB",
                f"{r['max_concurrent_sequences']} sequences",
            ]
            for r in gpu_results
        ],
        headers=["GPU Model", "Total VRAM", "10% Overhead", "4-bit Model", "VRAM for KV", "Max Concurrency"],
        tablefmt="github",
    ))

    # 5. Required GPUs for Heatwave Peak
    peak_analysis = calculate_peak_sessions_and_gpus(
        max_concurrent_per_gpu=gpu_results[0]["max_concurrent_sequences"],
        agent_latency_sec=7.85,
    )
    print("\n--- 5. REQUIRED GPUS FOR PEAK HEATWAVE LOAD ---")
    print(f"• Peak Heatwave Window QPM:        {peak_analysis['peak_qpm']} QPM")
    print(f"• Arrival Rate (lambda):           {peak_analysis['arrival_rate_per_sec']} queries/sec")
    print(f"• End-to-End Latency Target:       {peak_analysis['agent_latency_sec']} seconds")
    print(f"• Little's Law Active Sessions:    {peak_analysis['active_concurrent_sessions']} concurrent active queries")
    print(f"• Minimum GPUs Required (24GB):    {peak_analysis['gpus_required']} GPU (Capacity = {peak_analysis['max_concurrent_per_gpu']} concurrent)")
    print(f"• Recommended for High-Avail (HA): {peak_analysis['gpus_recommended_ha']} GPUs (Active-Passive or Load-Balanced)")

    # 6. Break-Even Analysis (Q*)
    be = run_break_even_analysis(
        gpu_monthly_cost_usd=248.20,
        gpu_capacity_concurrent=gpu_results[0]["max_concurrent_sequences"],
        cloud_api_cost_per_query=0.004575,
    )
    print("\n--- 6. BREAK-EVEN ANALYSIS (Q* Crossover Point) ---")
    print(f"• Real Cloud GPU Cost:             ${be['gpu_monthly_cost_usd']}/month per RTX 4090 (RunPod/Vast.ai @ $0.34/hr)")
    print(f"• Cloud API Cost per Query:        ${be['cloud_api_cost_per_query']:.6f} (GPT-4o-mini with prompt caching)")
    print(f"• Break-Even Daily Query Volume Q*: {be['q_star_daily']:.0f} queries/day")
    print(f"• Current Project Daily Traffic Q:  {Q_NORMAL} queries/day")
    print(f"• Assessment: Since current traffic ({Q_NORMAL} Q/day) is well above Q* ({be['q_star_daily']:.0f} Q/day), self-hosting is financially viable and saves operational budget.")

    print("\n--- STEP FUNCTION HOSTING VS CLOUD API COMPARISON ---")
    print(tabulate(
        [
            [
                r["daily_q"],
                f"{r['monthly_queries']:,}",
                r["gpus_needed"],
                f"${r['self_host_cost']:.2f}",
                f"${r['api_cost']:.2f}",
                r["cheaper"],
                f"${r['net_savings']:.2f}",
            ]
            for r in be["comparison_table"]
        ],
        headers=["Daily Q", "Monthly Queries", "GPUs Needed", "Self-Host Cost", "Cloud API Cost", "Cheaper Option", "Monthly Savings"],
        tablefmt="github",
    ))

    # Save summary
    out_dict = {
        "architecture": arch,
        "weight_memory": w_mem,
        "kv_cache_memory": kv_mem,
        "gpu_concurrency": gpu_results,
        "peak_analysis": peak_analysis,
        "break_even": be,
    }
    with open("kv_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print("\n[OK] KV and self-hosting summary saved to kv_summary.json")


if __name__ == "__main__":
    main()
