# Agentic AI — Assignment Beta
**Course**: Agentic AI (AI4015) — Fall 2026  
**Institution**: FAST-NUCES Karachi  
**Student**: Muhammad John Raza (Roll No: `23K-0069`)  

---

## Repository Structure

```
.
├── agentic q1/                                # Question 1: SaafHawa Agentic Assistant
│   ├── Assignment_Beta_Question_no_1_Report.md # Complete Academic Submission Report
│   ├── Assignment_Beta_Question_no_1_Calculations.xlsx # Multi-sheet Excel Model
│   ├── cost_plot.png                          # Self-Hosting vs. Cloud API Break-Even Plot
│   ├── model.py                               # Puter Live LLM Brain with Tool Schemas
│   ├── chat.py                                # Interactive Terminal Agent Interface
│   ├── tool.py                                # JSON Tool Schemas (AQI, Weather, Health KB)
│   ├── latencey.py                            # 96-call Empirical Latency Benchmark
│   ├── tool_latency_log.csv                   # Raw Empirical API Timing Log
│   ├── tool_latency_summary.json              # Statistical Summary (Mean, Median, P95)
│   ├── budget.py                              # 12-Model Cost & Prompt Caching Evaluation
│   ├── budget_summary.json                    # Workload & Monthly Cost Data
│   ├── kv.py                                  # Hugging Face config.json & KV Cache Math
│   ├── kv_summary.json                        # Weight Memory & GPU Concurrency Data
│   ├── config.json                            # Llama-3.1-8B-Instruct HF Config
│   ├── puter_brain.js                         # Node.js Puter SDK Runner
│   ├── package.json                           # Node dependencies
│   ├── requirements.txt                       # Python dependencies (uv pip)
│   └── .env.example                           # Template for Puter Auth Token
│
├── agentic q2/                                # Question 2: Parameter-Efficient Fine-Tuning
│   ├── Assignment_Beta_Question_no_2_23K_0069.ipynb # Submission Jupyter Notebook
│   ├── make_q2_notebook.py                    # Notebook Generation Engine
│   ├── colab_bridge_live.py                   # Google Colab MCP WebSocket Bridge
│   └── run_colab_mcp.py                       # Automated Colab MCP Launcher
│
├── chat.py                                    # Root Shortcut to Run SaafHawa Agent
├── requirements.txt                           # Root Project Dependencies
├── .gitignore                                 # Git Ignore Rules
└── README.md                                  # Repository Overview
```

---

## Quick Start

### 1. Environment Setup (uv)
```bash
# Create virtual environment
uv venv

# Activate on Windows
.venv\Scripts\activate

# Install dependencies
uv pip install -r "agentic q1/requirements.txt"
```

### 2. Configure Puter Auth Token
Create a `.env` file in `agentic q1/` with your token from [puter.com/dashboard](https://puter.com/dashboard):
```env
PUTER_AUTH_TOKEN=your_token_here
PUTER_MODEL=gpt-4o-mini
```

### 3. Run SaafHawa Agent (Question 1)
```bash
# Interactive conversation mode
python chat.py

# One-off query mode
python chat.py "Aj din 12 bajy Karachi Clifton mein temperature kitna hoga qk bachon ki school off honi hai?"
```

### 4. Run Question 2 PEFT Notebook
Open `agentic q2/Assignment_Beta_Question_no_2_23K_0069.ipynb` in [Google Colab](https://colab.research.google.com) on a T4 GPU and run all cells.
