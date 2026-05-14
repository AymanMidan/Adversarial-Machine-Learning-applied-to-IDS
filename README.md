# 🛡️ Adversarial IDS: Red Teaming & Blue Teaming a Neural Network Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Paper](https://img.shields.io/badge/Paper-UNDER%20REVIEW-orange)
![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red)

> **Can a deep learning‑based IDS be truly robust against realistic adversarial attacks?** > This project answers by building, breaking, and fixing an MLP intrusion detector on the UNSW‑NB15 dataset — using physically‑constrained adversarial examples and adversarial training.

## 🔥 Overview

Modern Intrusion Detection Systems (IDS) rely heavily on deep learning. But are they secure?  
**Spoiler: Not by default.**

We take a complete build‑break‑fix cycle:
- ✅ **Build** a high‑performance MLP‑based IDS (91.52% F1‑score)
- 🔴 **Red Team** it with two physically‑realistic attacks:
  - **CPGD** (white‑box, gradient‑based)
  - **PSO** (black‑box, evolutionary)
- 🔵 **Blue Team** it using adversarial training to gain cross‑attack robustness

**The result?** A model that resists its training attack (2.41% ASR) and significantly reduces the black‑box threat (from 97.95% → 49.61% ASR) — without sacrificing accuracy.

---

## ✨ Key Contributions

| Contribution | Description |
| :--- | :--- |
| 🧠 **Physically‑constrained attacks** | We respect network causality: you can’t reduce packet counts or session duration after the fact. Our projection operator $\mathcal{P}_c$ ensures every adversarial example is operationally realizable. |
| ⚡ **Black‑box beats white‑box** | PSO (97.95% ASR) almost doubles CPGD (53.23% ASR) — revealing dangerous non‑convex blind spots that gradient‑based attacks miss. |
| 🛡️ **Cross‑attack robustness** | Training only against CPGD cuts PSO’s success rate by half, proving that adversarial training generalizes beyond the attack seen. |
| 📊 **Complete pipeline** | From pre‑processing (42 numeric features, z‑score normalisation) to PyTorch training, attack generation, and evaluation — fully reproducible. |

---

## 📁 Dataset: UNSW‑NB15

We use the UNSW‑NB15 benchmark, a modern alternative to KDD’99 with realistic attack families.

| Property | Value |
| :--- | :--- |
| **Instances** | ~82,500 |
| **Features (selected)** | 42 (numeric only) |
| **Classes** | Binary (Benign=0, Attack=1) |
| **Attack families** | DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms, Backdoor, Analysis |
| **Class balance** | ~51% / 49% |

👉 **Pre‑processing:** remove non‑numeric fields (`proto`, `service`, `state`), standardise (z‑score on training set only), stratified 80/20 split.

---

## 🧠 Model Architecture (Baseline)

A shallow Multi‑Layer Perceptron (MLP) implemented in PyTorch:

$$
\begin{aligned}
\mathbf{h}_1 &= \text{ReLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 42} \\
\mathbf{h}_2 &= \text{ReLU}(\mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{32 \times 64} \\
\hat{y} &= \mathbf{w}_3^\top \mathbf{h}_2 + b_3
\end{aligned}
$$

- **Loss:** Binary Cross‑Entropy with Logits.
- **Optimizer:** Adam (lr=1e-3, weight decay=1e-4).
- **Hyperparameters:** Batch size = 256, Epochs = 20.

### Baseline performance (clean test set)

| Metric | Value |
| :--- | :--- |
| **Precision** | 98.50% |
| **Recall** | 85.46% |
| **F1‑Score** | 91.52% |

*Note: 14.5% of attacks are already missed by the vanilla model — a structural blind spot that adversarial attacks will ruthlessly exploit.*

---

## 🔴 Red Teaming: Offensive Adversarial ML

We generate adversarial examples $\mathbf{x}_{adv}$ that flip the model’s prediction from attack (1) to benign (0), while staying physically valid.

### 🔹 Constraint Projection $\mathcal{P}_c$ – The Game Changer

Unlike image attacks, network features cannot be arbitrarily modified. We partition features into two groups:
1. **Unilateral** (can only increase): `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `sload`, `dload` — because you cannot reduce past traffic.
2. **Free** (bidirectional): derived ratios, statistics.

The projection:

$$
[\mathcal{P}_c(\tilde{x})]_j = \begin{cases}
\max(\tilde{x}_j, x_j) & \text{if unilateral} \\
\text{clip}(\tilde{x}_j,\, x_j - \delta_j,\, x_j + \delta_j) & \text{otherwise}
\end{cases}
$$

*This makes our attacks operationally realistic — not just mathematical curiosities.*

### ⚔️ Attack 1: Constrained Projected Gradient Descent (CPGD)
**White‑box** (full model access, gradients). Iterative FGSM + projection:

$$
\begin{aligned}
\mathbf{g}^{(t)} &= \nabla_{\mathbf{x}} \mathcal{L}_{\text{BCE}}(f(\mathbf{x}^{(t)}), 1) \\
\tilde{\mathbf{x}}^{(t+1)} &= \mathbf{x}^{(t)} + \epsilon \cdot \operatorname{sign}(\mathbf{g}^{(t)}) \\
\mathbf{x}^{(t+1)} &= \mathcal{P}_c(\tilde{\mathbf{x}}^{(t+1)})
\end{aligned}
$$

| Metric | Value |
| :--- | :--- |
| **Attack Success Rate (ASR)** | **53.23%** |

*The model is fooled on more than one out of two attacks — a significant vulnerability.*

### 🐝 Attack 2: Particle Swarm Optimization (PSO)
**Black‑box** (only predictions, no gradients). A swarm of $N$ particles explores the constrained space, moving with inertia and social/cognitive components:

$$
\mathbf{v}_{i}^{(t+1)} = w\mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i^{(t)})
$$

| Metric | Value |
| :--- | :--- |
| **Attack Success Rate (ASR)** | **97.95%** |

*The black‑box swarm almost completely evades the IDS — a striking paradox: no gradients → higher success.*

### 📊 Comparison

| Attack | Paradigm | Knowledge | ASR | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **CPGD** | Gradient | White‑box | 53.23% | $\mathcal{O}(T \cdot d)$ |
| **PSO** | Swarm | Black‑box | 97.95% | $\mathcal{O}(T \cdot N \cdot d)$ |

**Why does black‑box work better?** The constrained gradient landscape is non‑convex and has “masked” gradients — local ascent gets stuck. PSO’s stochastic global search discovers adversarial basins that gradient‑based methods cannot reach.

---

## 🔵 Blue Teaming: Adversarial Training

We apply Madry’s min‑max formulation:

$$
\min_{\theta} \; \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \max_{\mathbf{x}' \in \mathcal{C}(\mathbf{x})} \mathcal{L}(f_{\theta}(\mathbf{x}'), y) \right]
$$

**Training loop** (5‑step fast CPGD per batch):
```python
for epoch in range(E):
    for X_batch, y_batch in dataloader:
        X_adv = cpgd_fast(X_batch, model)   # inner max
        
        X_total = torch.cat([X_batch, X_adv])
        y_total = torch.cat([y_batch, y_batch])
        
        loss = BCE_with_logits(model(X_total), y_total)
        loss.backward()
        optimizer.step()
