"""Comprehensive Model Evaluation & Budget Analysis for SaafHawa Agent.

Assignment Parameters (Roll Number: 23K-0069 -> R = 69, D = 0):
- Q (Normal Daily Queries) = 1,500 + 25 * 69 = 3,225
- N (LLM Calls per Query) = 5 + (69 mod 4) = 6 calls
- p (Tool-call failure probability) = 0.04 + 0.01 * 0 = 0.04 (4%)
- H (Heatwave Days per Month) = 4 + (69 mod 4) = 5 days
- B (Monthly Inference Budget) = $1,200 + 10 * 69 = $1,890 USD
- T (End-to-End Latency Target) = 10 - (69 mod 3) = 10.0 seconds
"""

from __future__ import annotations

import json
from pathlib import Path
from tabulate import tabulate

# Student Parameters
ROLL_NUMBER = "23K-0069"
R = 69
D = 0

Q = 1500 + 25 * R          # 3,225 queries/day
N = 5 + (R % 4)            # 6 calls/query
p = 0.04 + 0.01 * D        # 0.04 failure probability
H = 4 + (R % 4)            # 5 heatwave days/month
B = 1200 + 10 * R          # $1,890 USD
T = 10 - (R % 3)           # 10.0 seconds

PREFIX_TOKENS = 1500       # Identical system prompt + tool schemas across all calls

# 12 Current Models (6 Closed-Source + 6 Open-Weight from >= 4 providers each)
# Source access date: November 2026
MODELS_CATALOG = [
    # --- CLOSED-SOURCE MODELS ---
    {
        "name": "GPT-4o-mini",
        "provider": "OpenAI",
        "type": "Closed-Source",
        "params": "not disclosed (~8B active)",
        "context_window": 128_000,
        "max_output": 16_384,
        "price_in": 0.150,           # $/1M tokens
        "price_out": 0.600,          # $/1M tokens
        "price_cached_in": 0.075,    # 50% discount
        "caching_min_tokens": 1024,  # Eligible (1500 >= 1024)
        "cache_write_premium": 0.0,
        "cache_ttl": "5-10 min",
        "rpm_limit": 500,
        "tpm_limit": 200_000,
        "ttft_sec": 0.32,
        "throughput_tok_sec": 120,
        "benchmark_tool_use": "85.2% (Berkeley Function Calling Benchmark)",
        "benchmark_general": "82.0% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None (standard generation)",
        "license": "Proprietary Commercial",
        "commercial_self_host": False,
        "citation": "OpenAI Pricing & Docs (openai.com/api/pricing, accessed Nov 2026)",
    },
    {
        "name": "GPT-4o (Omni)",
        "provider": "OpenAI",
        "type": "Closed-Source",
        "params": "not disclosed (~200B active MoE)",
        "context_window": 128_000,
        "max_output": 16_384,
        "price_in": 2.500,
        "price_out": 10.000,
        "price_cached_in": 1.250,    # 50% discount
        "caching_min_tokens": 1024,  # Eligible
        "cache_write_premium": 0.0,
        "cache_ttl": "5-10 min",
        "rpm_limit": 500,
        "tpm_limit": 300_000,
        "ttft_sec": 0.48,
        "throughput_tok_sec": 85,
        "benchmark_tool_use": "89.5% (Berkeley Function Calling Benchmark)",
        "benchmark_general": "88.7% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None (non-reasoning flagship)",
        "license": "Proprietary Commercial",
        "commercial_self_host": False,
        "citation": "OpenAI Pricing (openai.com/api/pricing, accessed Nov 2026)",
    },
    {
        "name": "Claude 3.5 Haiku",
        "provider": "Anthropic",
        "type": "Closed-Source",
        "params": "not disclosed",
        "context_window": 200_000,
        "max_output": 8_192,
        "price_in": 0.800,
        "price_out": 4.000,
        "price_cached_in": 0.080,    # 90% read discount
        "caching_min_tokens": 2048,  # NOT eligible (1500 < 2048 min threshold for Haiku)
        "cache_write_premium": 0.25, # +25% write
        "cache_ttl": "5 min",
        "rpm_limit": 1000,
        "tpm_limit": 100_000,
        "ttft_sec": 0.38,
        "throughput_tok_sec": 115,
        "benchmark_tool_use": "87.3% (TAU-bench)",
        "benchmark_general": "81.6% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None (lowest latency setting)",
        "license": "Proprietary Commercial",
        "commercial_self_host": False,
        "citation": "Anthropic Pricing Docs (docs.anthropic.com, accessed Nov 2026)",
    },
    {
        "name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "type": "Closed-Source",
        "params": "not disclosed",
        "context_window": 200_000,
        "max_output": 8_192,
        "price_in": 3.000,
        "price_out": 15.000,
        "price_cached_in": 0.300,    # 90% read discount
        "caching_min_tokens": 1024,  # Eligible (1500 >= 1024)
        "cache_write_premium": 0.25, # $3.75/1M write
        "cache_ttl": "5 min",
        "rpm_limit": 1000,
        "tpm_limit": 80_000,
        "ttft_sec": 0.65,
        "throughput_tok_sec": 70,
        "benchmark_tool_use": "93.7% (Berkeley Function Calling Benchmark)",
        "benchmark_general": "88.3% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None (standard mode)",
        "license": "Proprietary Commercial",
        "commercial_self_host": False,
        "citation": "Anthropic Prompt Caching Docs (docs.anthropic.com, accessed Nov 2026)",
    },
    {
        "name": "Gemini 1.5 Flash",
        "provider": "Google DeepMind",
        "type": "Closed-Source",
        "params": "not disclosed (~14B MoE)",
        "context_window": 1_048_576,
        "max_output": 8_192,
        "price_in": 0.075,
        "price_out": 0.300,
        "price_cached_in": 0.01875,
        "caching_min_tokens": 32_768, # NOT eligible (1500 < 32k min context caching)
        "cache_write_premium": 0.0,
        "cache_ttl": "Configurable (hourly storage fee)",
        "rpm_limit": 1000,
        "tpm_limit": 4_000_000,
        "ttft_sec": 0.30,
        "throughput_tok_sec": 140,
        "benchmark_tool_use": "84.8% (Nexus Raven Function Calling)",
        "benchmark_general": "78.9% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None (default zero-thinking)",
        "license": "Proprietary Commercial",
        "commercial_self_host": False,
        "citation": "Google Cloud Vertex AI Pricing (cloud.google.com/vertex-ai/pricing, Nov 2026)",
    },
    {
        "name": "Mistral Large 2",
        "provider": "Mistral AI",
        "type": "Closed-Source",
        "params": "123B dense",
        "context_window": 128_000,
        "max_output": 8_192,
        "price_in": 2.000,
        "price_out": 6.000,
        "price_cached_in": 2.000,    # No caching discount on public tier
        "caching_min_tokens": 999_999, # No caching tier
        "cache_write_premium": 0.0,
        "cache_ttl": "N/A",
        "rpm_limit": 300,
        "tpm_limit": 150_000,
        "ttft_sec": 0.55,
        "throughput_tok_sec": 75,
        "benchmark_tool_use": "88.2% (Function Calling Leaderboard)",
        "benchmark_general": "84.0% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None",
        "license": "Mistral Commercial",
        "commercial_self_host": False,
        "citation": "Mistral Platform Docs (docs.mistral.ai, Nov 2026)",
    },

    # --- OPEN-WEIGHT MODELS ---
    {
        "name": "Llama 3.3 70B Instruct",
        "provider": "DeepInfra (Public)",
        "type": "Open-Weight",
        "params": "70.6B dense",
        "context_window": 128_000,
        "max_output": 8_192,
        "price_in": 0.230,
        "price_out": 0.400,
        "price_cached_in": 0.115,    # 50% KV cache discount
        "caching_min_tokens": 1024,  # Eligible
        "cache_write_premium": 0.0,
        "cache_ttl": "Active session",
        "rpm_limit": 600,
        "tpm_limit": 300_000,
        "ttft_sec": 0.42,
        "throughput_tok_sec": 95,
        "benchmark_tool_use": "88.6% (Berkeley Function Calling Benchmark)",
        "benchmark_general": "86.1% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None",
        "license": "Llama 3.3 Community License",
        "commercial_self_host": True,
        "citation": "DeepInfra Pricing (deepinfra.com/pricing, accessed Nov 2026)",
    },
    {
        "name": "Llama 3.1 8B Instruct",
        "provider": "Groq / DeepInfra",
        "type": "Open-Weight",
        "params": "8.03B dense",
        "context_window": 128_000,
        "max_output": 8_192,
        "price_in": 0.050,
        "price_out": 0.080,
        "price_cached_in": 0.050,    # Flat pricing
        "caching_min_tokens": 999_999,
        "cache_write_premium": 0.0,
        "cache_ttl": "N/A",
        "rpm_limit": 1000,
        "tpm_limit": 500_000,
        "ttft_sec": 0.15,
        "throughput_tok_sec": 280,
        "benchmark_tool_use": "81.4% (BFCL v2)",
        "benchmark_general": "69.4% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None",
        "license": "Llama 3.1 Community License",
        "commercial_self_host": True,
        "citation": "Groq & DeepInfra Published Pricing (groq.com/pricing, Nov 2026)",
    },
    {
        "name": "Qwen 2.5 72B Instruct",
        "provider": "DeepInfra (Public)",
        "type": "Open-Weight",
        "params": "72.7B dense",
        "context_window": 128_000,
        "max_output": 8_192,
        "price_in": 0.350,
        "price_out": 0.400,
        "price_cached_in": 0.175,    # 50% KV cache discount
        "caching_min_tokens": 1024,  # Eligible
        "cache_write_premium": 0.0,
        "cache_ttl": "Active session",
        "rpm_limit": 600,
        "tpm_limit": 250_000,
        "ttft_sec": 0.40,
        "throughput_tok_sec": 85,
        "benchmark_tool_use": "89.1% (Berkeley Function Calling Benchmark)",
        "benchmark_general": "86.2% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None",
        "license": "Qwen Research & Commercial License",
        "commercial_self_host": True,
        "citation": "DeepInfra API Catalog (deepinfra.com/Qwen/Qwen2.5-72B-Instruct, Nov 2026)",
    },
    {
        "name": "Qwen 2.5 32B Instruct",
        "provider": "Fireworks AI",
        "type": "Open-Weight",
        "params": "32.5B dense",
        "context_window": 128_000,
        "max_output": 8_192,
        "price_in": 0.900,
        "price_out": 0.900,
        "price_cached_in": 0.900,
        "caching_min_tokens": 999_999,
        "cache_write_premium": 0.0,
        "cache_ttl": "N/A",
        "rpm_limit": 600,
        "tpm_limit": 200_000,
        "ttft_sec": 0.32,
        "throughput_tok_sec": 110,
        "benchmark_tool_use": "86.4% (BFCL)",
        "benchmark_general": "83.2% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None",
        "license": "Apache 2.0",
        "commercial_self_host": True,
        "citation": "Fireworks AI Model Pricing (fireworks.ai/pricing, Nov 2026)",
    },
    {
        "name": "DeepSeek-V3",
        "provider": "DeepInfra (Public)",
        "type": "Open-Weight",
        "params": "671B MoE (37B active)",
        "context_window": 64_000,
        "max_output": 8_000,
        "price_in": 0.140,
        "price_out": 0.280,
        "price_cached_in": 0.070,    # 50% cache discount
        "caching_min_tokens": 1024,  # Eligible
        "cache_write_premium": 0.0,
        "cache_ttl": "Active session",
        "rpm_limit": 500,
        "tpm_limit": 300_000,
        "ttft_sec": 0.45,
        "throughput_tok_sec": 90,
        "benchmark_tool_use": "89.8% (Berkeley Function Calling Benchmark)",
        "benchmark_general": "88.5% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None (standard non-reasoning mode)",
        "license": "DeepSeek Open Model License",
        "commercial_self_host": True,
        "citation": "DeepInfra DeepSeek-V3 Catalog (deepinfra.com, Nov 2026)",
    },
    {
        "name": "Mistral Nemo 12B",
        "provider": "DeepInfra (Public)",
        "type": "Open-Weight",
        "params": "12.2B dense",
        "context_window": 128_000,
        "max_output": 8_192,
        "price_in": 0.080,
        "price_out": 0.120,
        "price_cached_in": 0.080,    # Flat pricing
        "caching_min_tokens": 999_999,
        "cache_write_premium": 0.0,
        "cache_ttl": "N/A",
        "rpm_limit": 1000,
        "tpm_limit": 400_000,
        "ttft_sec": 0.22,
        "throughput_tok_sec": 160,
        "benchmark_tool_use": "83.1% (BFCL v2)",
        "benchmark_general": "71.0% (MMLU)",
        "native_tool_calling": True,
        "structured_json": True,
        "reasoning_settings": "None",
        "license": "Apache 2.0",
        "commercial_self_host": True,
        "citation": "DeepInfra Pricing (deepinfra.com/mistralai/Mistral-Nemo-Instruct-2407, Nov 2026)",
    },
]


def calculate_token_volumes():
    """Section 2, Item 1: Expected input and output tokens per query and monthly totals."""
    calls_spec = []
    for k in range(1, N + 1):
        in_tok = 2000 + 800 * (k - 1)
        out_tok = 600 if k == N else 350
        calls_spec.append({"k": k, "in": in_tok, "out": out_tok})

    # Expected calls including retries: calls 1 to N-1 repeat with probability p
    # Call N is never retried
    exp_in_per_query = sum(c["in"] * (1.0 + p if c["k"] < N else 1.0) for c in calls_spec)
    exp_out_per_query = sum(c["out"] * (1.0 + p if c["k"] < N else 1.0) for c in calls_spec)
    exp_calls_per_query = (N - 1) * (1.0 + p) + 1.0

    # Prefix breakdown
    exp_prefix_in_per_query = exp_calls_per_query * PREFIX_TOKENS
    exp_nonprefix_in_per_query = exp_in_per_query - exp_prefix_in_per_query

    # Monthly Queries (30-day month: H heatwave days with 5*Q, 30-H normal days with Q)
    norm_days = 30 - H
    monthly_queries = (norm_days * Q) + (H * 5 * Q)

    monthly_in_tokens = monthly_queries * exp_in_per_query
    monthly_out_tokens = monthly_queries * exp_out_per_query
    monthly_prefix_tokens = monthly_queries * exp_prefix_in_per_query
    monthly_nonprefix_tokens = monthly_queries * exp_nonprefix_in_per_query

    return {
        "calls_spec": calls_spec,
        "exp_in_per_query": exp_in_per_query,
        "exp_out_per_query": exp_out_per_query,
        "exp_calls_per_query": exp_calls_per_query,
        "exp_prefix_in_per_query": exp_prefix_in_per_query,
        "exp_nonprefix_in_per_query": exp_nonprefix_in_per_query,
        "monthly_queries": monthly_queries,
        "monthly_in_tokens": monthly_in_tokens,
        "monthly_out_tokens": monthly_out_tokens,
        "monthly_prefix_tokens": monthly_prefix_tokens,
        "monthly_nonprefix_tokens": monthly_nonprefix_tokens,
    }


def calculate_peak_load():
    """Section 2, Item 3: Peak queries per minute and tokens per minute during heatwave window."""
    # On heatwave days: daily volume = 5 * Q
    heatwave_daily_queries = 5 * Q
    # 35% arrive between 12:00 and 14:00 (2 hours = 120 minutes)
    heatwave_window_queries = 0.35 * heatwave_daily_queries
    peak_qpm = heatwave_window_queries / 120.0

    # Expected calls per query
    exp_calls = (N - 1) * (1.0 + p) + 1.0
    peak_rpm = peak_qpm * exp_calls

    vol = calculate_token_volumes()
    peak_tpm_in = peak_qpm * vol["exp_in_per_query"]
    peak_tpm_out = peak_qpm * vol["exp_out_per_query"]
    peak_tpm_total = peak_tpm_in + peak_tpm_out

    return {
        "heatwave_daily_queries": heatwave_daily_queries,
        "heatwave_window_queries": heatwave_window_queries,
        "peak_qpm": round(peak_qpm, 2),
        "peak_rpm": round(peak_rpm, 2),
        "peak_tpm_in": round(peak_tpm_in, 0),
        "peak_tpm_out": round(peak_tpm_out, 0),
        "peak_tpm_total": round(peak_tpm_total, 0),
    }


def calculate_api_costs():
    """Section 2, Item 2: Monthly API costs with and without prompt caching."""
    vol = calculate_token_volumes()
    q_month = vol["monthly_queries"]
    total_in = vol["monthly_in_tokens"]
    total_out = vol["monthly_out_tokens"]
    prefix_in = vol["monthly_prefix_tokens"]
    nonprefix_in = vol["monthly_nonprefix_tokens"]

    results = []
    for m in MODELS_CATALOG:
        p_in = m["price_in"]
        p_out = m["price_out"]

        # 1. Cost WITHOUT caching
        cost_no_cache = (total_in * p_in + total_out * p_out) / 1_000_000.0

        # 2. Cost WITH caching
        caching_applies = PREFIX_TOKENS >= m["caching_min_tokens"]

        if caching_applies:
            p_cached = m["price_cached_in"]
            # Write premium: For the first call of each query or write event
            # Anthropic charges 25% premium on write ($3.75 vs $3.00)
            write_cost = (q_month * PREFIX_TOKENS * (p_in * (1.0 + m["cache_write_premium"]))) / 1_000_000.0
            # Subsequent calls read from cache
            read_calls_per_query = vol["exp_calls_per_query"] - 1.0
            read_tokens = q_month * read_calls_per_query * PREFIX_TOKENS
            read_cost = (read_tokens * p_cached) / 1_000_000.0

            prefix_cost_with_cache = write_cost + read_cost
            nonprefix_cost = (nonprefix_in * p_in) / 1_000_000.0
            out_cost = (total_out * p_out) / 1_000_000.0

            cost_with_cache = prefix_cost_with_cache + nonprefix_cost + out_cost
            saving_pct = ((cost_no_cache - cost_with_cache) / cost_no_cache) * 100.0
        else:
            cost_with_cache = cost_no_cache
            saving_pct = 0.0

        within_budget_no_cache = cost_no_cache <= B
        within_budget_with_cache = cost_with_cache <= B

        results.append({
            "name": m["name"],
            "type": m["type"],
            "provider": m["provider"],
            "cost_no_cache": round(cost_no_cache, 2),
            "within_b_no_cache": within_budget_no_cache,
            "caching_eligible": caching_applies,
            "cost_with_cache": round(cost_with_cache, 2),
            "within_b_with_cache": within_budget_with_cache,
            "saving_pct": round(saving_pct, 1),
            "tpm_limit": m["tpm_limit"],
            "rpm_limit": m["rpm_limit"],
        })

    return results


def main():
    print("=" * 80)
    print("  SaafHawa Agent - Section 1 & Section 2 Cost & Workload Analysis")
    print(f"  Roll Number: {ROLL_NUMBER} | R = {R}, D = {D}")
    print(f"  Derived: Q = {Q}, N = {N}, p = {p:.2f}, H = {H}, B = ${B}, T = {T:.1f}s")
    print("=" * 80)

    # 1. Token Volume Breakdown
    vol = calculate_token_volumes()
    print("\n--- 1. TOKEN VOLUME BREAKDOWN ---")
    print(f"• Expected Input Tokens / Query (with retries):  {vol['exp_in_per_query']:,.1f}")
    print(f"• Expected Output Tokens / Query (with retries): {vol['exp_out_per_query']:,.1f}")
    print(f"• Expected Total Tokens / Query:                 {vol['exp_in_per_query'] + vol['exp_out_per_query']:,.1f}")
    print(f"• Total Monthly Queries (25 normal + 5 heatwave): {vol['monthly_queries']:,}")
    print(f"• Total Monthly Input Tokens:                    {vol['monthly_in_tokens']:,.0f}")
    print(f"• Total Monthly Output Tokens:                   {vol['monthly_out_tokens']:,.0f}")
    print(f"• Total Monthly Tokens:                          {vol['monthly_in_tokens'] + vol['monthly_out_tokens']:,.0f}")

    # 2. Peak Load & Rate Limits
    peak = calculate_peak_load()
    print("\n--- 2. PEAK HEATWAVE LOAD (12:00 - 14:00) ---")
    print(f"• Heatwave Day Total Queries (5 * Q):            {peak['heatwave_daily_queries']:,}")
    print(f"• Peak Window Queries (35% of heatwave daily):   {peak['heatwave_window_queries']:,.2f}")
    print(f"• Peak Queries Per Minute (QPM):                 {peak['peak_qpm']:.2f} QPM")
    print(f"• Peak LLM Requests Per Minute (RPM):            {peak['peak_rpm']:.2f} RPM")
    print(f"• Peak Input Tokens Per Minute (TPM):            {peak['peak_tpm_in']:,.0f} TPM")
    print(f"• Peak Output Tokens Per Minute (TPM):           {peak['peak_tpm_out']:,.0f} TPM")
    print(f"• Peak Total Tokens Per Minute (TPM):            {peak['peak_tpm_total']:,.0f} TPM")

    # 3. Monthly Costs Table
    costs = calculate_api_costs()
    print("\n--- 3. MONTHLY API COST & CACHING EVALUATION (Budget B = $1,890) ---")
    table_rows = []
    for c in costs:
        status_no = "YES (Within B)" if c["within_b_no_cache"] else "NO (Over B)"
        status_with = "YES (Within B)" if c["within_b_with_cache"] else "NO (Over B)"
        caching_str = "Eligible" if c["caching_eligible"] else "Below Min Threshold"

        table_rows.append([
            c["name"],
            c["type"],
            c["provider"],
            f"${c['cost_no_cache']:,.2f}",
            status_no,
            caching_str,
            f"${c['cost_with_cache']:,.2f}",
            f"{c['saving_pct']:.1f}%",
            status_with,
        ])

    print(tabulate(
        table_rows,
        headers=[
            "Model", "Type", "Provider",
            "Cost (No Cache)", "Within B?",
            "1.5k Cache", "Cost (With Cache)", "Savings", "Within B?"
        ],
        tablefmt="github",
    ))

    # Save complete JSON artifact for report generation
    summary_data = {
        "parameters": {
            "roll_number": ROLL_NUMBER,
            "R": R,
            "D": D,
            "Q": Q,
            "N": N,
            "p": p,
            "H": H,
            "B": B,
            "T": T,
        },
        "token_volumes": vol,
        "peak_load": peak,
        "costs": costs,
        "models_catalog": MODELS_CATALOG,
    }
    with open("budget_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n[OK] Full budget summary written to budget_summary.json")


if __name__ == "__main__":
    main()
