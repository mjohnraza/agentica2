"""
Question 4: Sentiment Classification Evaluation & Justification Metrics
Student: Muhammad John Raza (23K-0069)
"""

import math
import json
import statistics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Ensure nltk wordnet / punkt if needed
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# -------------------------------------------------------------
# Confusion Matrices
# Rows = Actual [Positive, Negative, Neutral]
# Cols = Predicted [Positive, Negative, Neutral]
# -------------------------------------------------------------
classes = ["Positive", "Negative", "Neutral"]
total_samples = 300

cm_stage1 = [
    [100, 5, 15],  # Actual Pos: total 120
    [8, 80, 12],   # Actual Neg: total 100
    [20, 10, 50]   # Actual Neu: total 80
]

cm_stage2 = [
    [106, 4, 10],  # Actual Pos: total 120
    [6, 84, 10],   # Actual Neg: total 100
    [9, 7, 64]     # Actual Neu: total 80
]

def evaluate_cm(cm):
    n = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(len(cm)))
    acc = correct / n
    
    # 95% Confidence Interval for Accuracy
    # z = 1.959964
    z = 1.959964
    # Wald / Normal CI
    se_wald = math.sqrt((acc * (1.0 - acc)) / n)
    wald_low = acc - z * se_wald
    wald_high = acc + z * se_wald
    
    # Wilson Score CI
    denom = 1.0 + (z**2) / n
    center = (acc + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((acc * (1.0 - acc) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    wilson_low = center - margin
    wilson_high = center + margin
    
    # Per-class metrics
    per_class = {}
    row_totals = [sum(row) for row in cm]
    col_totals = [sum(cm[i][j] for i in range(len(cm))) for j in range(len(cm))]
    
    f1_list = []
    support_list = row_totals
    
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fp = col_totals[i] - tp
        fn = row_totals[i] - tp
        tn = n - tp - fp - fn
        
        prec = tp / col_totals[i] if col_totals[i] > 0 else 0.0
        rec = tp / row_totals[i] if row_totals[i] > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        per_class[c] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "support": row_totals[i],
            "pred_total": col_totals[i],
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        f1_list.append(f1)
        
    macro_f1 = sum(f1_list) / len(f1_list)
    weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / n
    micro_f1 = acc # for multi-class single-label classification
    
    return {
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "wald_ci": (wald_low, wald_high),
        "wilson_ci": (wilson_low, wilson_high),
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "micro_f1": micro_f1
    }

metrics_s1 = evaluate_cm(cm_stage1)
metrics_s2 = evaluate_cm(cm_stage2)

print("="*60)
print("STAGE 1 EVALUATION (Zero-Shot Baseline)")
print("="*60)
print(f"Overall Accuracy: {metrics_s1['accuracy']:.6f} ({metrics_s1['correct']}/{metrics_s1['n']})")
print(f"  95% Wald CI:   [{metrics_s1['wald_ci'][0]:.4f}, {metrics_s1['wald_ci'][1]:.4f}]")
print(f"  95% Wilson CI: [{metrics_s1['wilson_ci'][0]:.4f}, {metrics_s1['wilson_ci'][1]:.4f}]")
print("\nPer-class Metrics:")
for c in classes:
    p = metrics_s1['per_class'][c]
    print(f"  {c:10s} | Prec: {p['precision']:.4f} | Rec: {p['recall']:.4f} | F1: {p['f1']:.4f} | Support: {p['support']}")
print(f"\nAveraged F1 Metrics:")
print(f"  Macro-F1:    {metrics_s1['macro_f1']:.6f}")
print(f"  Weighted-F1: {metrics_s1['weighted_f1']:.6f}")
print(f"  Micro-F1:    {metrics_s1['micro_f1']:.6f}")

print("\n" + "="*60)
print("STAGE 2 EVALUATION (Improved Few-Shot + Neutral Definition)")
print("="*60)
print(f"Overall Accuracy: {metrics_s2['accuracy']:.6f} ({metrics_s2['correct']}/{metrics_s2['n']})")
print(f"  95% Wald CI:   [{metrics_s2['wald_ci'][0]:.4f}, {metrics_s2['wald_ci'][1]:.4f}]")
print(f"  95% Wilson CI: [{metrics_s2['wilson_ci'][0]:.4f}, {metrics_s2['wilson_ci'][1]:.4f}]")
print("\nPer-class Metrics:")
for c in classes:
    p = metrics_s2['per_class'][c]
    print(f"  {c:10s} | Prec: {p['precision']:.4f} | Rec: {p['recall']:.4f} | F1: {p['f1']:.4f} | Support: {p['support']}")
print(f"\nAveraged F1 Metrics:")
print(f"  Macro-F1:    {metrics_s2['macro_f1']:.6f}")
print(f"  Weighted-F1: {metrics_s2['weighted_f1']:.6f}")
print(f"  Micro-F1:    {metrics_s2['micro_f1']:.6f}")

# -------------------------------------------------------------
# Stability across 5 runs
# -------------------------------------------------------------
run_accuracies = [0.847, 0.840, 0.853, 0.837, 0.850]
mean_acc = statistics.mean(run_accuracies)
stdev_acc = statistics.stdev(run_accuracies)
cv_acc = (stdev_acc / mean_acc) * 100.0

print("\n" + "="*60)
print("PROMPT STABILITY ACROSS 5 RUNS")
print("="*60)
print(f"Runs: {run_accuracies}")
print(f"Mean Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
print(f"Sample Std:    {stdev_acc:.4f} ({stdev_acc*100:.2f}%)")
print(f"Mean +/- Std:  {mean_acc:.4f} +/- {stdev_acc:.4f} [{mean_acc-stdev_acc:.4f}, {mean_acc+stdev_acc:.4f}]")
print(f"Coeff of Var:  {cv_acc:.2f}%")

# -------------------------------------------------------------
# Justification Evaluation
# -------------------------------------------------------------
ref_str = "the delivery was late but the product quality is good"
cand_str = "the product quality is good but delivery was slow"

ref_tokens = ref_str.split()
cand_tokens = cand_str.split()

# BLEU-4
# Let's compute individual n-gram precisions
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

bleu_p = {}
for n in range(1, 5):
    cand_ng = get_ngrams(cand_tokens, n)
    ref_ng = get_ngrams(ref_tokens, n)
    match = 0
    ref_pool = list(ref_ng)
    for g in cand_ng:
        if g in ref_pool:
            match += 1
            ref_pool.remove(g)
    bleu_p[n] = (match, len(cand_ng), match / len(cand_ng) if len(cand_ng) > 0 else 0.0)

# Brevity penalty
r = len(ref_tokens)
c = len(cand_tokens)
bp = 1.0 if c > r else math.exp(1.0 - r / c)

# Standard sentence BLEU
smoothie = SmoothingFunction().method1
bleu_score_val = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

# ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(ref_str, cand_str)

# METEOR
meteor_val = meteor_score([ref_tokens], cand_tokens)

print("\n" + "="*60)
print("JUSTIFICATION NLP METRICS")
print("="*60)
print(f"Reference: '{ref_str}' (len={len(ref_tokens)})")
print(f"Candidate: '{cand_str}' (len={len(cand_tokens)})")
print("\nN-Gram Precisions:")
for n in range(1, 5):
    m, tot, prec = bleu_p[n]
    print(f"  P_{n}: {m}/{tot} = {prec:.4f}")
print(f"Brevity Penalty (BP): {bp:.4f} (c={c}, r={r})")
print(f"Sentence BLEU-4 (with method1 smoothing): {bleu_score_val:.4f}")

print("\nROUGE Scores (stemmed):")
for r_type in ['rouge1', 'rouge2', 'rougeL']:
    s = rouge_scores[r_type]
    print(f"  {r_type.upper():<8}: Precision={s.precision:.4f}, Recall={s.recall:.4f}, F1={s.fmeasure:.4f}")

print(f"\nMETEOR Score: {meteor_val:.4f}")

# -------------------------------------------------------------
# Perplexity Computation
# -------------------------------------------------------------
cand_token_probs = [0.60, 0.40, 0.50, 0.80, 0.70, 0.30, 0.25, 0.50, 0.20]
N = len(cand_token_probs)
log_probs = [math.log(p) for p in cand_token_probs]
sum_log_probs = sum(log_probs)
avg_nll = -sum_log_probs / N
perplexity = math.exp(avg_nll)

print("\n" + "="*60)
print("PERPLEXITY COMPUTATION")
print("="*60)
print(f"{'Token':<10} | {'Prob P(w)':<10} | {'ln P(w)':<12} | {'-ln P(w)':<12}")
print("-" * 50)
for tok, p, lp in zip(cand_tokens, cand_token_probs, log_probs):
    print(f"{tok:<10} | {p:<10.2f} | {lp:<12.4f} | {-lp:<12.4f}")
print("-" * 50)
print(f"Sum of log-probs:           {sum_log_probs:.6f}")
print(f"Average NLL (-1/N * sum):   {avg_nll:.6f}")
print(f"Perplexity PPL = exp(NLL):  {perplexity:.6f}")

# Save results for artifact generators
output_results = {
    "student_id": "23K-0069",
    "name": "Muhammad John Raza",
    "stage1": metrics_s1,
    "stage2": metrics_s2,
    "stability": {
        "runs": run_accuracies,
        "mean": mean_acc,
        "stdev": stdev_acc,
        "cv": cv_acc
    },
    "nlp_metrics": {
        "bleu_p": {str(k): list(v) for k, v in bleu_p.items()},
        "bleu_score": bleu_score_val,
        "brevity_penalty": bp,
        "rouge1": {"precision": rouge_scores['rouge1'].precision, "recall": rouge_scores['rouge1'].recall, "f1": rouge_scores['rouge1'].fmeasure},
        "rouge2": {"precision": rouge_scores['rouge2'].precision, "recall": rouge_scores['rouge2'].recall, "f1": rouge_scores['rouge2'].fmeasure},
        "rougeL": {"precision": rouge_scores['rougeL'].precision, "recall": rouge_scores['rougeL'].recall, "f1": rouge_scores['rougeL'].fmeasure},
        "meteor": meteor_val
    },
    "perplexity": {
        "tokens": cand_tokens,
        "probs": cand_token_probs,
        "log_probs": log_probs,
        "sum_log_probs": sum_log_probs,
        "avg_nll": avg_nll,
        "ppl": perplexity
    }
}

with open("agentic q4/q4_results.json", "w") as f:
    json.dump(output_results, f, indent=2)

print("\nAll Question 4 results successfully exported to agentic q4/q4_results.json")
