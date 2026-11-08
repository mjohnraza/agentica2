"""
Generator for Question 3 Jupyter Notebook and Markdown Report
Student: Muhammad John Raza (23K-0069)
"""

import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FAST-NUCES Karachi — Fall 2026\n",
                "## AI4015: Agentic AI — Assignment Beta\n",
                "### Question 3: Knowledge Distillation Calculations & Optimization Step\n",
                "**Student Name:** Muhammad John Raza  \n",
                "**Roll Number:** `23K-0069`  \n",
                "**Derived Parameters:** $R = 69$, $D = 0$  \n",
                "---\n",
                "### Problem Formulation\n",
                "Under the knowledge distillation scenario, a large teacher model ($z_t$) and a smaller student model ($z_s$) are predicting the next word for the prompt:\n",
                "> *\"The patient was admitted to the ___\"*\n",
                "\n",
                "Vocabulary ($K = 4$ candidate words):\n",
                "1. `hospital` (Ground Truth target, $y = [1, 0, 0, 0]$)\n",
                "2. `clinic`\n",
                "3. `ward`\n",
                "4. `school`\n",
                "\n",
                "Given Logits:\n",
                "- **Teacher Logits:** $z_t = [5.0, 3.0, 2.0, 0.0]$\n",
                "- **Student Logits:** $z_s = [3.0, 2.5, 1.0, 0.5]$\n",
                "\n",
                "Hyperparameters:\n",
                "- Distillation Temperature: $T = 2$\n",
                "- Mixing Weight: $\\alpha = 0.5$\n",
                "- Learning Rate: $\\eta = 0.5$\n",
                "\n",
                "Loss Function (Hinton et al., 2015):\n",
                "$$L(z_s) = \\alpha \\cdot T^2 \\cdot \\text{KL}(p_T \\parallel q_T) + (1 - \\alpha) \\cdot \\text{CE}(y, q_1)$$\n",
                "where:\n",
                "- $p_T = \\text{softmax}(z_t / T)$\n",
                "- $q_T = \\text{softmax}(z_s / T)$\n",
                "- $q_1 = \\text{softmax}(z_s / 1)$\n",
                "- Natural logarithm (base $e$) is used throughout."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import math\n",
                "\n",
                "# Candidate vocabulary and given vectors\n",
                "vocab = ['hospital', 'clinic', 'ward', 'school']\n",
                "y = [1.0, 0.0, 0.0, 0.0]\n",
                "z_t = [5.0, 3.0, 2.0, 0.0]\n",
                "z_s = [3.0, 2.5, 1.0, 0.5]\n",
                "T = 2.0\n",
                "alpha = 0.5\n",
                "eta = 0.5\n",
                "\n",
                "def softmax(logits, temp=1.0):\n",
                "    scaled = [x / temp for x in logits]\n",
                "    max_val = max(scaled)\n",
                "    exps = [math.exp(x - max_val) for x in scaled]\n",
                "    sum_exps = sum(exps)\n",
                "    return [e / sum_exps for e in exps]\n",
                "\n",
                "print('Environment initialized successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Part (a): Probabilities at $T = 1$\n",
                "For standard temperature $T = 1$:\n",
                "$$p_{1,i} = \\frac{\\exp(z_{t,i})}{\\sum_{j=1}^4 \\exp(z_{t,j})}, \\quad q_{1,i} = \\frac{\\exp(z_{s,i})}{\\sum_{j=1}^4 \\exp(z_{s,j})}$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "p_1 = softmax(z_t, temp=1.0)\n",
                "q_1 = softmax(z_s, temp=1.0)\n",
                "\n",
                "print(f\"{'Word':<10} | {'Teacher z_t':<12} | {'Teacher p_1':<14} | {'Student z_s':<12} | {'Student q_1':<14}\")\n",
                "print('-'*70)\n",
                "for i, word in enumerate(vocab):\n",
                "    print(f\"{word:<10} | {z_t[i]:<12.1f} | {p_1[i]:<14.6f} | {z_s[i]:<12.1f} | {q_1[i]:<14.6f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Part (b): Softened Probabilities at $T = 2$\n",
                "Scaling logits by temperature $T = 2$:\n",
                "- Teacher scaled logits: $z_t / 2 = [2.5, 1.5, 1.0, 0.0]$\n",
                "- Student scaled logits: $z_s / 2 = [1.5, 1.25, 0.5, 0.25]$\n",
                "\n",
                "$$p_{T,i} = \\frac{\\exp(z_{t,i}/2)}{\\sum_{j=1}^4 \\exp(z_{t,j}/2)}, \\quad q_{T,i} = \\frac{\\exp(z_{s,i}/2)}{\\sum_{j=1}^4 \\exp(z_{s,j}/2)}$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "p_T = softmax(z_t, temp=T)\n",
                "q_T = softmax(z_s, temp=T)\n",
                "\n",
                "print(f\"{'Word':<10} | {'Teacher z_t/2':<14} | {'Teacher p_T':<14} | {'Student z_s/2':<14} | {'Student q_T':<14}\")\n",
                "print('-'*72)\n",
                "for i, word in enumerate(vocab):\n",
                "    print(f\"{word:<10} | {z_t[i]/T:<14.2f} | {p_T[i]:<14.6f} | {z_s[i]/T:<14.2f} | {q_T[i]:<14.6f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Part (c): KL Divergence $\\text{KL}(p_T \\parallel q_T)$\n",
                "Kullback-Leibler divergence between softened teacher distribution $p_T$ and student distribution $q_T$:\n",
                "$$\\text{KL}(p_T \\parallel q_T) = \\sum_{i=1}^4 p_{T,i} \\ln \\left(\\frac{p_{T,i}}{q_{T,i}}\\right) = \\sum_{i=1}^4 \\left[ p_{T,i} \\ln(p_{T,i}) - p_{T,i} \\ln(q_{T,i}) \\right]$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "kl_terms = [p * math.log(p / q) for p, q in zip(p_T, q_T)]\n",
                "kl_div = sum(kl_terms)\n",
                "\n",
                "print(f\"{'Word':<10} | {'p_T':<12} | {'q_T':<12} | {'p_T/q_T':<12} | {'p_T * ln(p_T/q_T)':<18}\")\n",
                "print('-'*70)\n",
                "for i, word in enumerate(vocab):\n",
                "    print(f\"{word:<10} | {p_T[i]:<12.6f} | {q_T[i]:<12.6f} | {p_T[i]/q_T[i]:<12.6f} | {kl_terms[i]:<+18.6f}\")\n",
                "print('-'*70)\n",
                "print(f\"Total KL Divergence KL(p_T || q_T) = {kl_div:.6f} nats\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Part (d): Hard-Label Cross-Entropy Loss $\\text{CE}(y, q_1)$\n",
                "Since $y = [1, 0, 0, 0]$ (word 1: `hospital`):\n",
                "$$\\text{CE}(y, q_1) = -\\sum_{i=1}^4 y_i \\ln(q_{1,i}) = -\\ln(q_{1,1})$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "ce_loss = -math.log(q_1[0])\n",
                "print(f\"q_1[hospital] = {q_1[0]:.6f}\")\n",
                "print(f\"CE(y, q_1)    = -ln({q_1[0]:.6f}) = {ce_loss:.6f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Part (e): Total Distillation Loss $L$\n",
                "Using the Hinton et al. formulation:\n",
                "$$L = \\alpha \\cdot T^2 \\cdot \\text{KL}(p_T \\parallel q_T) + (1 - \\alpha) \\cdot \\text{CE}(y, q_1)$$\n",
                "With $T = 2 \\implies T^2 = 4$, and $\\alpha = 0.5$:\n",
                "$$L_{\\text{KD}} = 0.5 \\times 4 \\times \\text{KL} = 2.0 \\times 0.081607 = 0.163215$$\n",
                "$$L_{\\text{CE}} = 0.5 \\times \\text{CE} = 0.5 \\times 0.601005 = 0.300502$$\n",
                "$$L = 0.163215 + 0.300502 = 0.463717$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loss_kd_scaled = (T ** 2) * kl_div\n",
                "loss_kd_weighted = alpha * loss_kd_scaled\n",
                "loss_ce_weighted = (1.0 - alpha) * ce_loss\n",
                "total_loss = loss_kd_weighted + loss_ce_weighted\n",
                "\n",
                "print(f\"Distillation Component (alpha * T^2 * KL) = {alpha} * {T**2} * {kl_div:.6f} = {loss_kd_weighted:.6f}\")\n",
                "print(f\"Hard-label Component   ((1-alpha) * CE)   = {1.0-alpha} * {ce_loss:.6f}       = {loss_ce_weighted:.6f}\")\n",
                "print(f\"Total Loss L                              = {total_loss:.6f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Part (f): Analytical Gradient $\\frac{\\partial L}{\\partial z_s}$\n",
                "#### Derivation:\n",
                "1. For the hard cross-entropy component:\n",
                "   $$\\frac{\\partial \\text{CE}(y, q_1)}{\\partial z_{s,i}} = q_{1,i} - y_i$$\n",
                "\n",
                "2. For the scaled KL distillation component $L_{\\text{KD}} = T^2 \\text{KL}(p_T \\parallel q_T) = -T^2 \\sum_k p_{T,k} \\ln q_{T,k} + \\text{const}$:\n",
                "   Let $v_k = z_{s,k} / T$, so $q_{T,k} = \\text{softmax}(v)_k$.\n",
                "   $$\\frac{\\partial \\left( -\\sum_k p_{T,k} \\ln q_{T,k} \\right)}{\\partial v_i} = q_{T,i} - p_{T,i}$$\n",
                "   By the chain rule, $\\frac{\\partial v_i}{\\partial z_{s,i}} = \\frac{1}{T}$:\n",
                "   $$\\frac{\\partial \\text{KL}}{\\partial z_{s,i}} = \\frac{1}{T} (q_{T,i} - p_{T,i})$$\n",
                "   Multiplying by $T^2$:\n",
                "   $$\\frac{\\partial (T^2 \\text{KL})}{\\partial z_{s,i}} = T^2 \\cdot \\frac{1}{T} (q_{T,i} - p_{T,i}) = T (q_{T,i} - p_{T,i})$$\n",
                "\n",
                "3. Combining with weights $\\alpha$ and $(1 - \\alpha)$:\n",
                "   $$\\frac{\\partial L}{\\partial z_{s,i}} = \\alpha \\cdot T \\cdot (q_{T,i} - p_{T,i}) + (1 - \\alpha) \\cdot (q_{1,i} - y_i)$$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "grad_kd = [alpha * T * (q - p) for q, p in zip(q_T, p_T)]\n",
                "grad_ce = [(1.0 - alpha) * (q - yi) for q, yi in zip(q_1, y)]\n",
                "grad_total = [g_kd + g_ce for g_kd, g_ce in zip(grad_kd, grad_ce)]\n",
                "\n",
                "# Numerical verification via central finite differences\n",
                "def evaluate_loss(test_logits):\n",
                "    q_T_eval = softmax(test_logits, temp=T)\n",
                "    q_1_eval = softmax(test_logits, temp=1.0)\n",
                "    kl_eval = sum(p * math.log(p / q) for p, q in zip(p_T, q_T_eval))\n",
                "    ce_eval = -math.log(q_1_eval[0])\n",
                "    return alpha * (T ** 2) * kl_eval + (1.0 - alpha) * ce_eval\n",
                "\n",
                "eps = 1e-6\n",
                "grad_numerical = []\n",
                "for i in range(len(z_s)):\n",
                "    zp = list(z_s); zp[i] += eps\n",
                "    zm = list(z_s); zm[i] -= eps\n",
                "    grad_numerical.append((evaluate_loss(zp) - evaluate_loss(zm)) / (2.0 * eps))\n",
                "\n",
                "print(f\"{'Word':<10} | {'Grad KD':<12} | {'Grad CE':<12} | {'Analytical dL/dz':<18} | {'Numerical dL/dz':<18} | {'Abs Diff'}\")\n",
                "print('-'*90)\n",
                "for i, word in enumerate(vocab):\n",
                "    diff = abs(grad_total[i] - grad_numerical[i])\n",
                "    print(f\"{word:<10} | {grad_kd[i]:<+12.6f} | {grad_ce[i]:<+12.6f} | {grad_total[i]:<+18.6f} | {grad_numerical[i]:<+18.6f} | {diff:.2e}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "### Part (g): Parameter Update & Recomputed Loss\n",
                "Applying one gradient descent step with learning rate $\\eta = 0.5$:\n",
                "$$z_s' = z_s - \\eta \\cdot \\nabla_{z_s} L$$\n",
                "\n",
                "We then recompute the probabilities and loss after the single optimization step."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "z_s_new = [z - eta * g for z, g in zip(z_s, grad_total)]\n",
                "q_1_new = softmax(z_s_new, temp=1.0)\n",
                "q_T_new = softmax(z_s_new, temp=T)\n",
                "\n",
                "kl_new = sum(p * math.log(p / q) for p, q in zip(p_T, q_T_new))\n",
                "ce_new = -math.log(q_1_new[0])\n",
                "total_loss_new = alpha * (T ** 2) * kl_new + (1.0 - alpha) * ce_new\n",
                "\n",
                "print(f\"{'Word':<10} | {'Old z_s':<10} | {'Gradient':<12} | {'Delta (-0.5*g)':<16} | {'New z_s':<12} | {'Old q_1':<10} | {'New q_1':<10}\")\n",
                "print('-'*88)\n",
                "for i, word in enumerate(vocab):\n",
                "    delta = -eta * grad_total[i]\n",
                "    print(f\"{word:<10} | {z_s[i]:<10.4f} | {grad_total[i]:<+12.6f} | {delta:<+16.6f} | {z_s_new[i]:<12.6f} | {q_1[i]:<10.6f} | {q_1_new[i]:<10.6f}\")\n",
                "\n",
                "print('-'*88)\n",
                "print(f\"Initial Loss L(z_s)  = {total_loss:.6f}\")\n",
                "print(f\"Updated Loss L(z_s') = {total_loss_new:.6f}\")\n",
                "print(f\"Loss Reduction       = {total_loss - total_loss_new:.6f} ({((total_loss - total_loss_new)/total_loss)*100:.2f}% improvement)\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("agentic q3/Assignment_Beta_Question_no_3_23K_0069.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Saved agentic q3/Assignment_Beta_Question_no_3_23K_0069.ipynb")
