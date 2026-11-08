"""
Question 3: Knowledge Distillation Calculations
Student Roll Number: 23K-0069
Student Name: Muhammad John Raza
"""

import math
import json

def softmax(logits, temp=1.0):
    scaled = [x / temp for x in logits]
    max_val = max(scaled)
    exps = [math.exp(x - max_val) for x in scaled]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# Given values
vocab = ["hospital", "clinic", "ward", "school"]
y = [1.0, 0.0, 0.0, 0.0]
z_t = [5.0, 3.0, 2.0, 0.0]
z_s = [3.0, 2.5, 1.0, 0.5]
T = 2.0
alpha = 0.5
eta = 0.5

print("================ QUESTION 3 DETAILED COMPUTATIONS ================")

# (a) Probabilities at T=1
p_1 = softmax(z_t, temp=1.0)
q_1 = softmax(z_s, temp=1.0)

print("\n--- (a) Probabilities at T = 1 ---")
for i, word in enumerate(vocab):
    print(f"  {word:10s} | Teacher z_t={z_t[i]:4.1f} -> p_1={p_1[i]:.6f} | Student z_s={z_s[i]:4.1f} -> q_1={q_1[i]:.6f}")

# (b) Softened probabilities at T=2
p_T = softmax(z_t, temp=T)
q_T = softmax(z_s, temp=T)

print(f"\n--- (b) Softened Probabilities at T = {T:.1f} ---")
for i, word in enumerate(vocab):
    print(f"  {word:10s} | Teacher z_t/2={z_t[i]/T:4.2f} -> p_T={p_T[i]:.6f} | Student z_s/2={z_s[i]/T:4.2f} -> q_T={q_T[i]:.6f}")

# (c) KL Divergence KL(p_T || q_T)
kl_terms = [p * math.log(p / q) for p, q in zip(p_T, q_T)]
kl_div = sum(kl_terms)

print(f"\n--- (c) KL Divergence KL(p_T || q_T) ---")
for i, word in enumerate(vocab):
    print(f"  {word:10s}: p_T={p_T[i]:.6f}, q_T={q_T[i]:.6f}, term={kl_terms[i]:.6f}")
print(f"  Total KL(p_T || q_T) = {kl_div:.6f}")

# (d) Hard-label Cross-Entropy Loss CE(y, q_1)
ce_loss = -sum(y_i * math.log(q_i) for y_i, q_i in zip(y, q_1) if y_i > 0)
print(f"\n--- (d) Cross-Entropy Loss CE(y, q_1) ---")
print(f"  CE(y, q_1) = -ln(q_1[0]) = -ln({q_1[0]:.6f}) = {ce_loss:.6f}")

# (e) Total Distillation Loss
# Hinton's standard formula: L = alpha * T^2 * KL + (1 - alpha) * CE
loss_kd_scaled = (T ** 2) * kl_div
total_loss = alpha * loss_kd_scaled + (1.0 - alpha) * ce_loss

print(f"\n--- (e) Total Distillation Loss L ---")
print(f"  alpha * T^2 * KL = {alpha} * {T**2} * {kl_div:.6f} = {alpha * loss_kd_scaled:.6f}")
print(f"  (1 - alpha) * CE = {1 - alpha} * {ce_loss:.6f} = {(1 - alpha) * ce_loss:.6f}")
print(f"  Total Loss L = {total_loss:.6f}")

# (f) Gradient of Loss w.r.t Student Logits z_s
# Analytical gradient: dL/dz_s = alpha * T * (q_T - p_T) + (1 - alpha) * (q_1 - y)
grad_kd = [alpha * T * (q - p) for q, p in zip(q_T, p_T)]
grad_ce = [(1.0 - alpha) * (q - yi) for q, yi in zip(q_1, y)]
grad_total = [g_kd + g_ce for g_kd, g_ce in zip(grad_kd, grad_ce)]

print(f"\n--- (f) Gradient dL/dz_s ---")
print("  Word       q_T - p_T    alpha*T*(q_T-p_T)    q_1 - y    (1-a)*(q_1-y)    dL/dz_s")
for i, word in enumerate(vocab):
    print(f"  {word:10s} {q_T[i]-p_T[i]:+10.6f} {grad_kd[i]:+18.6f} {q_1[i]-y[i]:+10.6f} {grad_ce[i]:+16.6f} {grad_total[i]:+10.6f}")

# Check with numerical finite-difference gradient verification
def compute_loss(logits):
    q_T_local = softmax(logits, temp=T)
    q_1_local = softmax(logits, temp=1.0)
    kl_local = sum(p * math.log(p / q) for p, q in zip(p_T, q_T_local))
    ce_local = -sum(y_i * math.log(q_i) for y_i, q_i in zip(y, q_1_local) if y_i > 0)
    return alpha * (T ** 2) * kl_local + (1.0 - alpha) * ce_local

eps = 1e-6
grad_numerical = []
for i in range(len(z_s)):
    z_plus = list(z_s)
    z_plus[i] += eps
    z_minus = list(z_s)
    z_minus[i] -= eps
    grad_numerical.append((compute_loss(z_plus) - compute_loss(z_minus)) / (2.0 * eps))

print(f"\n  Numerical Finite-Difference Gradient Verification (eps={eps}):")
for i, word in enumerate(vocab):
    print(f"  {word:10s} Analytical: {grad_total[i]:.6f} | Numerical: {grad_numerical[i]:.6f} | Diff: {abs(grad_total[i]-grad_numerical[i]):.2e}")

# (g) One step gradient descent update: z_s' = z_s - eta * grad
z_s_new = [z - eta * g for z, g in zip(z_s, grad_total)]

# Recompute loss after update
q_1_new = softmax(z_s_new, temp=1.0)
q_T_new = softmax(z_s_new, temp=T)
kl_new = sum(p * math.log(p / q) for p, q in zip(p_T, q_T_new))
ce_new = -sum(y_i * math.log(q_i) for y_i, q_i in zip(y, q_1_new) if y_i > 0)
total_loss_new = alpha * (T ** 2) * kl_new + (1 - alpha) * ce_new

print(f"\n--- (g) Parameter Update (eta = {eta}) and New Loss ---")
print("  Word       z_s (old)    Gradient    Update (-eta*g)    z_s (new)    q_1 (old)    q_1 (new)")
for i, word in enumerate(vocab):
    print(f"  {word:10s} {z_s[i]:10.4f} {grad_total[i]:+10.6f} {-eta*grad_total[i]:+16.6f} {z_s_new[i]:12.6f} {q_1[i]:10.6f} {q_1_new[i]:10.6f}")

print(f"\n  Old Loss: {total_loss:.6f}")
print(f"  New Loss: {total_loss_new:.6f}")
print(f"  Loss Reduction: {total_loss - total_loss_new:.6f} ({(total_loss - total_loss_new)/total_loss*100:.2f}%)")

# Save all results to JSON for artifact generation
results = {
    "student_id": "23K-0069",
    "name": "Muhammad John Raza",
    "vocab": vocab,
    "ground_truth": y,
    "z_t": z_t,
    "z_s": z_s,
    "T": T,
    "alpha": alpha,
    "eta": eta,
    "p_1": p_1,
    "q_1": q_1,
    "p_T": p_T,
    "q_T": q_T,
    "kl_terms": kl_terms,
    "kl_div": kl_div,
    "ce_loss": ce_loss,
    "total_loss": total_loss,
    "grad_kd": grad_kd,
    "grad_ce": grad_ce,
    "grad_total": grad_total,
    "z_s_new": z_s_new,
    "q_1_new": q_1_new,
    "q_T_new": q_T_new,
    "kl_new": kl_new,
    "ce_new": ce_new,
    "total_loss_new": total_loss_new,
    "loss_reduction": total_loss - total_loss_new
}

with open("agentic q3/q3_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults successfully written to agentic q3/q3_results.json")
