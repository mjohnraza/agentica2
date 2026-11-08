# Agentic AI — Assignment Beta
**Course**: Agentic AI (AI4015) — Fall 2026  
**Institution**: FAST-NUCES Karachi  
**Student**: Muhammad John Raza (Roll No: `23K-0069`)  
**Derived Parameters**: $R = 69$, $D = 0$  

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
├── agentic q3/                                # Question 3: Knowledge Distillation Mechanics
│   ├── Assignment_Beta_Question_no_3_23K_0069.md   # Rigorous Step-by-Step Derivation Report
│   ├── Assignment_Beta_Question_no_3_23K_0069.ipynb # Interactive Jupyter Notebook
│   ├── solve_q3.py                            # Closed-form Math & Numerical Gradient Checker
│   ├── make_q3_artifacts.py                   # Artifact Generator
│   └── q3_results.json                        # Exact JSON Export of Quantities & Gradients
│
├── agentic q4/                                # Question 4: Sentiment Evaluation & Justification
│   ├── Assignment_Beta_Question_no_4_Report.md     # Academic Evaluation Report
│   ├── Assignment_Beta_Question_no_4_23K_0069.ipynb # Submission Jupyter Notebook
│   ├── solve_q4.py                            # Statistical & NLP Evaluation Engine
│   ├── make_q4_artifacts.py                   # Notebook Generation Script
│   └── q4_results.json                        # Complete Metrics Export (BLEU, ROUGE, PPL, CI)
│
├── agentic q5/                                # Question 5: Research Paper Summary
│   └── Assignment_Beta_Question_no_5_Report.md # 2-Page Structured Summary (Ersoy & Erşahin)
│
├── chat.py                                    # Root Shortcut to Run SaafHawa Agent
├── requirements.txt                           # Root Project Dependencies
├── .gitignore                                 # Git Ignore Rules
└── README.md                                  # Repository Overview
```

---

## Question Overview & Key Findings

### Question 1: SaafHawa Agentic Assistant (`agentic q1/`)
- **System**: Live multi-turn AI assistant providing real-time Karachi environmental intelligence in Roman Urdu.
- **Tools**: Live Open-Meteo Air Quality API, Weather Forecast API, English $\to$ Roman Urdu translator, and localized medical advice vector knowledge base.
- **Empirical Latency**: 96 live API requests logged (`median = 64.76 ms`, `mean = 76.84 ms`).
- **Cost Budgeting**: Evaluated 12 models across monthly query volume ($Q_{\text{month}} = 161,250$ queries) and peak load ($47.03\text{ QPM}$).
- **Hardware Sizing**: Extracted Hugging Face `config.json` for Llama-3.1-8B-Instruct. Sized FP16 KV-cache for 6,600 tokens ($825\text{ MiB}$) with 21 concurrent sequences on RTX 4090 ($24\text{ GB}$).
- **Break-Even Analysis**: Self-hosting on RTX 4090 beats Cloud API (GPT-4o-mini) at $Q^* = 1,085\text{ queries/day}$ ($32,550\text{ queries/month}$).

### Question 2: Parameter-Efficient Fine-Tuning (`agentic q2/`)
- **Notebook**: `Assignment_Beta_Question_no_2_23K_0069.ipynb`
- **Theory**: $W_0 \in \mathbb{R}^{4096 \times 4096}$ with $r=16 \implies 131,072$ parameters ($0.78125\%$ trainable); mathematical proof of zero initial adaptation perturbation ($B = 0 \implies \Delta W = 0$).
- **Benchmark**: Evaluated Baseline vs. LoRA vs. QLoRA vs. DoRA vs. AdaLoRA on `Qwen2.5-1.5B-Instruct` across 2,000 function-calling examples (1,800 train / 200 test).
- **Rank Scaling**: Evaluated $r \in \{4, 8, 16, 32\}$ on DoRA, demonstrating diminishing returns beyond $r = 8$.
- **Production Recommendation**: DoRA ($r=8$) selected for shipping. Detailed failure mode analysis and multi-agent adapter hot-swapping risks.

### Question 3: Knowledge Distillation Mechanics (`agentic q3/`)
- **Notebook**: `Assignment_Beta_Question_no_3_23K_0069.ipynb`
- **Report**: `Assignment_Beta_Question_no_3_23K_0069.md`
- **Setup**: Context: *"The patient was admitted to the ___"*, $K=4$ candidate vocabulary (`hospital`, `clinic`, `ward`, `school`), ground truth $y = [1, 0, 0, 0]^T$.
  - Teacher Logits: $z_t = [5.0, 3.0, 2.0, 0.0]^T$
  - Student Logits: $z_s = [3.0, 2.5, 1.0, 0.5]^T$
  - Hyperparameters: $T = 2.0$, $\alpha = 0.5$, $\eta = 0.5$.
- **Key Quantities**:
  - $T=1$ Probabilities: Teacher $p_{1,1} = 0.839025$, Student $q_{1,1} = 0.548260$.
  - $T=2$ Softened Probabilities: Teacher $p_{T,1} = 0.597695$, Student $q_{T,1} = 0.410984$.
  - Softened KL Divergence: $\text{KL}(p_T \parallel q_T) = 0.081607\text{ nats}$.
  - Hard Cross-Entropy Loss: $\text{CE}(y, q_1) = -\ln(0.548260) = 0.601005\text{ nats}$.
  - Total Loss: $L = 0.5(4)(0.081607) + 0.5(0.601005) = \mathbf{0.463717}$.
  - Analytical Gradients: $\nabla_{z_s} L = [-0.412581, +0.266463, +0.054928, +0.091189]^T$ (verified against central finite differences to $10^{-11}$).
  - Parameter Update: Updated logits $z_s' = [3.206290, 2.366768, 0.972536, 0.454405]^T$.
  - Post-Step Loss: $L' = \mathbf{0.349256}$ ($24.68\%$ relative loss reduction in a single step).

### Question 4: Sentiment Classification & Justification (`agentic q4/`)
- **Notebook**: `Assignment_Beta_Question_no_4_23K_0069.ipynb`
- **Report**: `Assignment_Beta_Question_no_4_Report.md`
- **Dataset**: 300 customer reviews (Positive: 120, Negative: 100, Neutral: 80).
- **Stage 1 (Zero-Shot Baseline)**:
  - Accuracy: $76.67\%$ ($230/300$). 95% Wilson Score CI: $[71.56\%, 81.10\%]$, Wald CI: $[71.88\%, 81.45\%]$.
  - Per-class $F_1$: Pos $= 0.8065$, Neg $= 0.8205$, Neu $= 0.6369$.
  - Averaged $F_1$: Macro $= 0.7546$, Weighted $= 0.7659$, Micro $= 0.7667$.
  - Error Diagnosis: Neutral collapsed ($37.5\%$ error rate) due to polarizing mixed sentiment without explicit guidance.
- **Stage 2 (Few-Shot + Explicit Neutral Definition)**:
  - Accuracy: $84.67\%$ ($254/300$, $\mathbf{+8.00\%}$ gain). 95% Wilson CI: $[80.15\%, 88.30\%]$.
  - Per-class $F_1$: Pos $= 0.8797$, Neg $= 0.8615$, Neu $= \mathbf{0.7805}$ ($\mathbf{+22.55\%}$ relative increase).
  - Averaged $F_1$: Macro $= 0.8406$, Weighted $= 0.8472$, Micro $= 0.8467$.
- **Prompt Stability**: Evaluated across 5 random exemplar permutations:
  - Runs: $[0.847, 0.840, 0.853, 0.837, 0.850] \implies \text{Mean} = 84.54\%, s = 0.67\%, CV = 0.79\%$.
- **Justification NLP Metrics**:
  - Reference: *"the delivery was late but the product quality is good"*
  - Candidate: *"the product quality is good but delivery was slow"*
  - Scores: Sentence BLEU-4 $= 0.4750$ ($\text{BP} = 0.8948$), ROUGE-1 $F_1 = 0.8421$, ROUGE-2 $F_1 = 0.5882$, ROUGE-L $F_1 = 0.5263$, METEOR $= \mathbf{0.7868}$.
- **Perplexity**: Candidate string perplexity $\text{PPL} = \mathbf{2.3249}$ (Average NLL $= 0.8437\text{ nats/token}$).

### Question 5: Cross-Domain RAG Paper Summary (`agentic q5/`)
- **Report**: `Assignment_Beta_Question_no_5_Report.md`
- **Paper**: *"A Comparative Evaluation of RAG Architectures for Cross-Domain LLM Applications: Design, Implementation, and Assessment"* by Pınar Ersoy & M. Erşahin (*IEEE Access, Vol. 13*).
- **Core Insights**:
  - Systematically evaluated Sparse (BM25), Dense, Hybrid (RRF), and Fusion RAG across Finance and E-Commerce.
  - Highlights failure of pure dense embeddings on alphanumeric IDs, temporal tickers, and financial tables ($28\%$ error rate).
  - Demonstrates that **Hybrid RAG with Reciprocal Rank Fusion (RRF)** provides the superior enterprise Pareto frontier ($0.89$ faithfulness with only $1.25\times$ latency overhead).

---

## Quick Start & Verification

### 1. Environment Setup (uv)
```bash
# Create and activate virtual environment
uv venv
.venv\Scripts\activate

# Install all project dependencies
uv pip install -r requirements.txt
```

### 2. Run Question Scripts
```bash
# Question 1: Run Interactive SaafHawa Agent
python chat.py

# Question 3: Run Knowledge Distillation Solver & Gradient Verification
python "agentic q3/solve_q3.py"

# Question 4: Run Sentiment Evaluation & NLP Metrics
python "agentic q4/solve_q4.py"
```

---

## Academic Integrity & Tool Declarations
All theoretical derivations, empirical benchmarks, and code implementations were developed and verified by **Muhammad John Raza (23K-0069)** for the FAST-NUCES AI4015 course.
