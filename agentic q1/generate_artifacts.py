"""Generates the required Cost Plot and Excel Workbook for Question 1 submission."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load summaries
with open("budget_summary.json", encoding="utf-8") as f:
    budget_data = json.load(f)

with open("kv_summary.json", encoding="utf-8") as f:
    kv_data = json.load(f)

# 1. Generate High-Quality Cost Plot
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

daily_q_range = np.linspace(100, 15000, 500)
cloud_api_cost_per_q = kv_data["break_even"]["cloud_api_cost_per_query"]
monthly_queries_range = (25 * daily_q_range) + (5 * 5 * daily_q_range)  # 50 * daily_q
cloud_cost = monthly_queries_range * cloud_api_cost_per_q

gpu_monthly = kv_data["break_even"]["gpu_monthly_cost_usd"]
gpu_cap = 21

# Step function for self-hosting
self_host_cost = []
for q in daily_q_range:
    p_qpm = (0.35 * 5 * q) / 120.0
    active = int(np.ceil((p_qpm / 60.0) * 7.85))
    gpus = max(1, int(np.ceil(active / gpu_cap)))
    self_host_cost.append(gpus * gpu_monthly)
self_host_cost = np.array(self_host_cost)

ax.plot(
    daily_q_range,
    cloud_cost,
    label="Closed-Source Cloud API (GPT-4o-mini with Prompt Caching)",
    color="#1f77b4",
    linewidth=2.5,
)
ax.step(
    daily_q_range,
    self_host_cost,
    label="Self-Hosting Llama 3.1 8B (RTX 4090 Cloud @ $248.20/GPU)",
    color="#d62728",
    linewidth=2.5,
    where="post",
)

# Markers for Q* and Project Q
q_star = kv_data["break_even"]["q_star_daily"]
crossover_cost = q_star * 50 * cloud_api_cost_per_q
ax.scatter(
    [q_star],
    [crossover_cost],
    color="#2ca02c",
    s=120,
    zorder=5,
    label=f"Break-Even Q* = {q_star:,.0f} queries/day ($248.20/mo)",
)

project_q = budget_data["parameters"]["Q"]
project_cloud_cost = project_q * 50 * cloud_api_cost_per_q
ax.scatter([project_q], [project_cloud_cost], color="#1f77b4", s=100, zorder=5)
ax.scatter([project_q], [gpu_monthly], color="#d62728", s=100, zorder=5)

ax.axvline(
    x=project_q,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Project Traffic Q = {project_q:,} queries/day",
)
ax.axhline(
    y=budget_data["parameters"]["B"],
    color="purple",
    linestyle=":",
    linewidth=1.8,
    label=f"Monthly Budget B = ${budget_data['parameters']['B']:,}",
)

ax.annotate(
    f"Current Operating Point:\nSelf-Host = ${gpu_monthly:.2f}/mo\nAPI = ${project_cloud_cost:.2f}/mo\nSavings = ${project_cloud_cost - gpu_monthly:.2f}/mo (66.4%)",
    xy=(project_q, (project_cloud_cost + gpu_monthly) / 2),
    xytext=(project_q + 1000, (project_cloud_cost + gpu_monthly) / 2 + 300),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffcc", alpha=0.9),
    fontsize=9,
)

ax.set_title(
    "SaafHawa Agent: Monthly Cost vs. Daily Query Volume (Break-Even Analysis)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Daily Query Volume (Q) [queries/day]", fontsize=11, fontweight="bold")
ax.set_ylabel("Total Monthly Infrastructure Cost (USD)", fontsize=11, fontweight="bold")
ax.set_xlim(0, 15000)
ax.set_ylim(0, 4000)
ax.legend(loc="upper left", frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig("cost_plot.png")
plt.close()
print("[OK] Successfully generated cost_plot.png")

# 2. Generate Excel Spreadsheet
with pd.ExcelWriter("Assignment_Beta_Question_no_1_Calculations.xlsx", engine="openpyxl") as writer:
    # Sheet 1: Parameters & Tokens
    df_params = pd.DataFrame([
        {"Parameter": "Roll Number", "Symbol": "Roll", "Formula": "Student ID", "Value": budget_data["parameters"]["roll_number"], "Unit": "ID"},
        {"Parameter": "Last Two Digits", "Symbol": "R", "Formula": "Last 2 digits", "Value": budget_data["parameters"]["R"], "Unit": "integer"},
        {"Parameter": "Digit Sum Mod 10", "Symbol": "D", "Formula": "Sum(digits) mod 10", "Value": budget_data["parameters"]["D"], "Unit": "integer"},
        {"Parameter": "Normal Daily Queries", "Symbol": "Q", "Formula": "1,500 + 25 * R", "Value": budget_data["parameters"]["Q"], "Unit": "queries/day"},
        {"Parameter": "Calls per Query", "Symbol": "N", "Formula": "5 + (R mod 4)", "Value": budget_data["parameters"]["N"], "Unit": "calls"},
        {"Parameter": "Tool Failure Probability", "Symbol": "p", "Formula": "0.04 + 0.01 * D", "Value": budget_data["parameters"]["p"], "Unit": "probability"},
        {"Parameter": "Heatwave Days per Month", "Symbol": "H", "Formula": "4 + (R mod 4)", "Value": budget_data["parameters"]["H"], "Unit": "days"},
        {"Parameter": "Monthly Inference Budget", "Symbol": "B", "Formula": "1,200 + 10 * R", "Value": budget_data["parameters"]["B"], "Unit": "USD"},
        {"Parameter": "Latency Target", "Symbol": "T", "Formula": "10 - (R mod 3)", "Value": budget_data["parameters"]["T"], "Unit": "seconds"},
        {"Parameter": "Expected Input Tokens / Query", "Symbol": "Exp_In", "Formula": "Sum(In_k * (1+p))", "Value": budget_data["token_volumes"]["exp_in_per_query"], "Unit": "tokens"},
        {"Parameter": "Expected Output Tokens / Query", "Symbol": "Exp_Out", "Formula": "Sum(Out_k * (1+p))", "Value": budget_data["token_volumes"]["exp_out_per_query"], "Unit": "tokens"},
        {"Parameter": "Total Monthly Queries", "Symbol": "Q_month", "Formula": "(25*Q) + (5*5*Q)", "Value": budget_data["token_volumes"]["monthly_queries"], "Unit": "queries"},
        {"Parameter": "Total Monthly Input Tokens", "Symbol": "In_month", "Formula": "Q_month * Exp_In", "Value": budget_data["token_volumes"]["monthly_in_tokens"], "Unit": "tokens"},
        {"Parameter": "Total Monthly Output Tokens", "Symbol": "Out_month", "Formula": "Q_month * Exp_Out", "Value": budget_data["token_volumes"]["monthly_out_tokens"], "Unit": "tokens"},
    ])
    df_params.to_excel(writer, sheet_name="Parameters_Tokens", index=False)

    # Sheet 2: Model Costs & Prompt Caching
    df_costs = pd.DataFrame(budget_data["costs"])
    df_costs.to_excel(writer, sheet_name="Model_Costs_Caching", index=False)

    # Sheet 3: Peak Heatwave Workload
    df_peak = pd.DataFrame([budget_data["peak_load"]])
    df_peak.to_excel(writer, sheet_name="Peak_Heatwave_Load", index=False)

    # Sheet 4: KV Cache & GPU Concurrency
    df_kv = pd.DataFrame([
        {"Architecture Parameter": "Model Name", "Value": kv_data["architecture"]["model_name"]},
        {"Architecture Parameter": "Total Parameters", "Value": "8.03 Billion"},
        {"Architecture Parameter": "Hidden Layers", "Value": kv_data["architecture"]["num_layers"]},
        {"Architecture Parameter": "Attention Heads", "Value": kv_data["architecture"]["num_heads"]},
        {"Architecture Parameter": "Key-Value Heads", "Value": kv_data["architecture"]["num_kv_heads"]},
        {"Architecture Parameter": "Hidden Size", "Value": kv_data["architecture"]["hidden_size"]},
        {"Architecture Parameter": "Head Dimension", "Value": kv_data["architecture"]["head_dim"]},
        {"Architecture Parameter": "Weight Memory (FP16)", "Value": str(kv_data["weight_memory"]["fp16_gb"]) + " GB"},
        {"Architecture Parameter": "Weight Memory (INT8)", "Value": str(kv_data["weight_memory"]["int8_gb"]) + " GB"},
        {"Architecture Parameter": "Weight Memory (INT4)", "Value": str(kv_data["weight_memory"]["int4_gb"]) + " GB"},
        {"Architecture Parameter": "Longest Context (Tokens)", "Value": kv_data["kv_cache_memory"]["sequence_length"]},
        {"Architecture Parameter": "KV Cache per Sequence", "Value": str(kv_data["kv_cache_memory"]["kv_mib"]) + " MiB (" + str(kv_data["kv_cache_memory"]["kv_gb"]) + " GB)"},
        {"Architecture Parameter": "RTX 4090 Max Concurrent Sequences", "Value": kv_data["gpu_concurrency"][0]["max_concurrent_sequences"]},
        {"Architecture Parameter": "A100 Max Concurrent Sequences", "Value": kv_data["gpu_concurrency"][1]["max_concurrent_sequences"]},
        {"Architecture Parameter": "Peak Heatwave Active Sessions (Little s Law)", "Value": kv_data["peak_analysis"]["active_concurrent_sessions"]},
        {"Architecture Parameter": "Minimum GPUs Required (24GB)", "Value": kv_data["peak_analysis"]["gpus_required"]},
    ])
    df_kv.to_excel(writer, sheet_name="KV_Cache_GPU", index=False)

    # Sheet 5: Break-Even Analysis
    df_be = pd.DataFrame(kv_data["break_even"]["comparison_table"])
    df_be.to_excel(writer, sheet_name="Break_Even_Analysis", index=False)

print("[OK] Successfully generated Assignment_Beta_Question_no_1_Calculations.xlsx")

