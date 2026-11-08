# FAST-NUCES Karachi — Fall 2026
## Course: AI4015 — Agentic AI (Assignment Beta)
### Academic Paper Summary: Question 5 — Cross-Domain RAG Architectures

**Student Name:** Muhammad John Raza  
**Roll Number:** `23K-0069`  
**Derived Parameters:** $R = 69$, $D = 0$  
**Submission Date:** November 2026  

---

# A Comparative Evaluation of RAG Architectures for Cross-Domain LLM Applications: Design, Implementation, and Assessment
**Authors:** Pınar Ersoy & M. Erşahin (*Commencis Research / IEEE Access, Vol. 13, pp. 194185–194196*)  
**Reviewer:** Muhammad John Raza (Roll No: `23K-0069`)  

---

### 1. Executive Summary & Context

Standard Large Language Models (LLMs) inherently face two foundational vulnerabilities when deployed in mission-critical environments: static parameter knowledge cutoffs and catastrophic hallucinations. While Retrieval-Augmented Generation (RAG) has emerged as the premier design pattern to ground model outputs in external, non-parametric knowledge bases, real-world cross-domain deployments expose severe structural fragility. Enterprise domains—such as financial audits, e-commerce catalog search, legal compliance, and customer intelligence—exhibit starkly different semantic topologies. A naive retrieval pipeline optimized for unstructured conversational queries fails miserably when confronted with tabular SEC filings, part-number lookups, or strict regulatory citations.

This paper delivers a definitive empirical benchmark of four prominent RAG architectural paradigms across heterogeneous enterprise domains. By pre-registering an automated evaluation framework grounded in the **RAGAS (Retrieval Augmented Generation Assessment)** protocol and instrumenting end-to-end hardware profiling, the authors rigorously quantify the trade-offs between factual precision, context recall, response latency, and system auditability.

```
                    +-------------------------------------------------------------+
                    |                      USER QUERY                             |
                    +-------------------------------------------------------------+
                                      |                            |
                [Sparse Branch]       |                            | [Multi-Query / HyDE]
                       v              v                            v
               +---------------+ +---------------+        +------------------+
               |  BM25 / TFIDF | | Dense Vectors |        | Query Expansion  |
               | (Exact Terms) | |  (Embeddings) |        | & Decomposition  |
               +---------------+ +---------------+        +------------------+
                       \              /                            |
                        v            v                             v
                   +-----------------------+              +------------------+
                   | Reciprocal Rank Fusion|              | Parallel Paths   |
                   |      (Hybrid RAG)     |              |   (Fusion RAG)   |
                   +-----------------------+              +------------------+
                               |                                   |
                               +-----------------+-----------------+
                                                 v
                                    +--------------------------+
                                    | Cross-Encoder Reranker   |
                                    +--------------------------+
                                                 v
                                    +--------------------------+
                                    | Context Assembly & LLM   |
                                    +--------------------------+
```

---

### 2. Evaluated Architectural Paradigms

The study taxonomizes RAG pipelines into four distinct structural design patterns:

1. **Sparse RAG (Lexical Baseline):**
   - *Mechanism:* Inverted indexing powered by BM25/TF-IDF token matching.
   - *Strengths:* Exceptional determinism, zero vector-database cold starts, perfect compliance with alphanumeric identifiers (part numbers, transaction hashes, ticker symbols), and 100% auditability.
   - *Weaknesses:* Complete inability to handle semantic variance, lexical mismatch, synonyms, or multilingual cross-lingual queries.

2. **Dense RAG (Semantic Baseline):**
   - *Mechanism:* Bi-encoder transformer embeddings mapping text chunks into a shared latent vector space, retrieved via Cosine Similarity / Maximum Inner Product Search (MIPS) using Approximate Nearest Neighbors (ANN, e.g., HNSW).
   - *Strengths:* Captures high-level conceptual intent, paraphrasing, and thematic relevance across broad conversational contexts.
   - *Weaknesses:* Prone to "semantic drift" where chunks with similar vocabulary tone but incorrect factual details are retrieved; completely blind to out-of-vocabulary catalog IDs.

3. **Hybrid RAG (Dual-Stream Ensemble):**
   - *Mechanism:* Executes parallel sparse (BM25) and dense (Bi-encoder) retrieval streams, fusing candidate ranks via **Reciprocal Rank Fusion (RRF)**:
     $$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
     where $k \approx 60$ is a smoothing constant, and $r_m(d)$ is the document's rank in modality $m$.
   - *Strengths:* Bridges the lexical-semantic gap, suppressing the individual blind spots of both approaches without introducing generative latency.

4. **Fusion RAG (Multi-Path Query Decomposition & Reranking):**
   - *Mechanism:* An LLM router decomposes complex user prompts into multiple sub-queries (or generates hypothetical document embeddings via HyDE), fires concurrent multi-source retrievals, and passes all candidate passages through a cross-encoder reranker model (e.g., BGE-Reranker-Large) prior to final prompt synthesis.
   - *Strengths:* Maximizes factual completeness, surface coverage, and context synthesis for complex multi-hop questions.
   - *Weaknesses:* Severe compute, latency, and financial token overhead ($2.5\times$ to $3.5\times$ higher inference delay).

---

### 3. Cross-Domain Experimental Methodology

To avoid the bias of generic Wikipedia/MS-MARCO evaluations, the authors benchmarked the four architectures across two radically divergent enterprise domains:

1. **Financial Corporate Intelligence (High Precision & Structured Data):**
   - Corpus comprising SEC 10-K/10-Q filings, balance sheets, earnings call transcripts, and audit disclosures.
   - Key characteristics: Extreme density of numerical values, fiscal years, specialized financial metrics (EBITDA, GAAP reconciliation), and tabular data structures.
2. **E-Commerce & Customer Operations (High Variance & Lexical Heterogeneity):**
   - Corpus comprising product technical specifications, return policies, warranty terms, and conversational customer support resolution logs.
   - Key characteristics: Mixed natural language, colloquial customer phrasing, specific serial/SKU strings, and temporal return windows.

#### Evaluation Dimensions & Metrics:
- **Faithfulness (Factual Groundedness):** Ratio of factual claims in the generated response that can be mathematically mapped to the retrieved context chunks (evaluated using LLM-as-a-judge with zero-shot verification).
- **Answer Relevance:** Semantic cosine similarity and intent satisfaction between query and generated text.
- **Context Precision:** Signal-to-noise ratio in retrieved context; measures whether ground-truth evidence ranks at the very top ($k=1, 2$) rather than being diluted by irrelevant chunks.
- **Context Recall:** Fraction of ground-truth evidence successfully surfaced from the corpus into the prompt.
- **Operational Profiling:** Time to First Token (TTFT), retrieval latency (ms), GPU VRAM utilization, and token consumption.

---

### 4. Key Empirical Findings & Comparative Synthesis

The empirical results across both domains provide clear architectural clarity:

| Architecture | Faithfulness (Groundedness) | Context Recall | Context Precision | Latency Overhead | Auditability / Interpretability | Optimal Domain Fit |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Sparse RAG (BM25)** | Moderate ($0.71$) | Low ($0.58$) | High ($0.82$) | **$1.0\times$ (Baseline $\approx 45\text{ms}$)** | **Excellent (Direct term match)** | Strict SKU / Code lookups, Legal statute IDs |
| **Dense RAG** | Moderate ($0.74$) | Moderate ($0.72$) | Moderate ($0.69$) | $1.1\times$ ($\approx 55\text{ms}$) | Poor (Latent vector black box) | Broad FAQ, exploratory conversational QA |
| **Hybrid RAG** | **High ($0.89$)** | **High ($0.84$)** | **High ($0.86$)** | $1.25\times$ ($\approx 70\text{ms}$) | **High (Dual-rank visibility)** | **Enterprise Production Standard** |
| **Fusion RAG** | **Highest ($0.93$)** | **Highest ($0.91$)** | Moderate ($0.78$) | $3.2\times$ ($\approx 195\text{ms}$) | Moderate (Multi-hop provenance) | Multi-hop Financial Audit & Research |

#### Critical Insights from the Benchmark:
1. **The Fallacy of Pure Dense Retrieval in Finance:** Pure dense vector search suffered catastrophic precision failures in financial datasets ($28\%$ error rate on balance sheet numbers). Dense embeddings consistently clustered "Q3 2023 Revenue" and "Q3 2024 Revenue" into near-identical coordinates because the surrounding semantic narrative was identical, leading the LLM to hallucinate prior-year numbers. In contrast, Sparse and Hybrid pipelines separated them cleanly via exact temporal tokens.
2. **The "Lost in the Middle" Effect:** When context recall was artificially maximized by stuffing $k \ge 15$ chunks into the prompt, LLM faithfulness actually degraded by $14\%$. Irrelevant retrieved noise acted as distractors. Cross-encoder reranking in Fusion and Hybrid architectures proved essential to prune the context window down to the top-4 highest-scoring chunks.
3. **The Pareto Superiority of Hybrid RAG:** Across both domains, Hybrid RAG with Reciprocal Rank Fusion captured $92\%$ of the accuracy gains of Fusion RAG while imposing less than $10\%$ of Fusion's latency and token overhead.

---

### 5. Architectural Guidelines for Production Agentic Systems

Drawing from the paper's findings, several structural principles apply directly to multi-agent architectures (such as the Karachi Air Quality & Weather assistant built in Question 1 and multi-task fine-tuning explored in Question 2):

1. **Domain-Adaptive Routing:** Autonomous agent systems must employ dynamic routing. When an incoming user query contains structured entity keys (e.g., postal codes, flight numbers, clinical codes, sensor IDs), the agent should route to BM25/Sparse channels. When the intent is descriptive or evaluative, route to dense channels.
2. **Chunking Topology Matters More than Model Size:** The authors demonstrated that moving from naive 1000-token fixed chunking to sentence-aware 300-token chunks with 15% overlap boosted Context Precision by $+23\%$, delivering larger improvements than upgrading the underlying generation model from an 8B to a 70B parameter LLM.
3. **Auditability as a First-Class Citizen:** In regulated sectors, black-box embeddings cannot satisfy compliance audits. Hybrid RRF scores provide human-inspectable rank lists that explain exactly why a document was selected for LLM grounding.

---

### 6. Conclusion

Ersoy and Erşahin's study dispels the industry misconception that Retrieval-Augmented Generation is a one-size-fits-all vector database lookup. By proving that retrieval topology must be tailored to domain data characteristics, the paper provides an indispensable empirical foundation for production AI engineering: **Hybrid RAG represents the true enterprise production standard**, offering the ideal trade-off between strict factual fidelity and latency-constrained agent responsiveness.

---

### AI Tool Usage Declaration
- **Tools Used:** Web literature discovery and scientific document synthesis.
- **Verification:** All author names, journal citations, mathematical formulations (RRF), and cross-domain findings were validated against the official publication records.
