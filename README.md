<div align="center">

# 🛡️ Adversarial IDS
### Red Teaming & Blue Teaming a Neural Network Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-UNDER%20REVIEW-orange)]()
[![Made with](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red)]()

> *Can a deep learning‑based IDS be truly robust against realistic adversarial attacks?*
> This project answers by **building**, **breaking**, and **fixing** an MLP intrusion detector on the UNSW‑NB15 dataset — using physically‑constrained adversarial examples and adversarial training.

</div>

---

## 🔥 Overview

Modern Intrusion Detection Systems (IDS) rely heavily on deep learning. But are they secure?
**Spoiler: Not by default.**

We take a complete **build‑break‑fix** cycle:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   ✅  BUILD     ────▶│   🔴 RED TEAM   ────▶│   🔵 BLUE TEAM 
│                 │     │                 │     │                 │
│  MLP-based IDS  │     │  CPGD  │  PSO   │     │  Adversarial    │
│  F1 = 91.52%    │     │ 53.23% │ 97.95% │     │  Training       │
│                 │     │  ASR   │  ASR   │     │  CPGD → 2.41%   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- ✅ **Build** a high‑performance MLP‑based IDS (91.52% F1‑score)
- 🔴 **Red Team** it with two physically‑realistic attacks: **CPGD** (white‑box) and **PSO** (black‑box)
- 🔵 **Blue Team** it using adversarial training to gain cross‑attack robustness

The result? A model that resists its training attack (**2.41% ASR**) and significantly reduces the black‑box threat (**97.95% → 49.61% ASR**) — without sacrificing accuracy.

---

## ✨ Key Contributions

| Contribution | Description |
|:-------------|:------------|
| 🧠 **Physically‑constrained attacks** | We respect network causality: you can't reduce packet counts or session duration after the fact. Our projection operator $\mathcal{P}_c$ ensures every adversarial example is operationally realizable. |
| ⚡ **Black‑box beats white‑box** | PSO (97.95% ASR) almost doubles CPGD (53.23% ASR) — revealing dangerous non‑convex blind spots that gradient‑based attacks miss. |
| 🛡️ **Cross‑attack robustness** | Training only against CPGD cuts PSO's success rate by half, proving that adversarial training generalizes beyond the attack seen. |
| 📊 **Complete pipeline** | From pre‑processing (42 numeric features, z‑score normalisation) to PyTorch training, attack generation, and evaluation — fully reproducible. |

---

## 📁 Dataset: UNSW‑NB15

We use the **UNSW‑NB15** benchmark, a modern alternative to KDD'99 with realistic attack families.

| Property | Value |
|:---------|:------|
| Instances | ~82,500 |
| Features (selected) | 42 (numeric only) |
| Classes | Binary (Benign=0, Attack=1) |
| Attack families | DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms, Backdoor, Analysis |
| Class balance | ~51% / 49% |

👉 **Pre‑processing:** remove non‑numeric fields (`proto`, `service`, `state`), standardise (z‑score on training set only), stratified 80/20 split.

---

## 🧠 Model Architecture (Baseline)

A shallow **Multi‑Layer Perceptron (MLP)** implemented in PyTorch:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 42}$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{32 \times 64}$$
$$\hat{y} = \mathbf{w}_3^\top \mathbf{h}_2 + b_3$$

- **Loss:** Binary Cross‑Entropy with Logits
- **Optimizer:** Adam (lr=1e-3, weight decay=1e-4)
- **Batch size** = 256, **Epochs** = 20

```
Input (d=42)  →  [Linear 64 | ReLU]  →  [Linear 32 | ReLU]  →  [Linear 1]  →  σ(ŷ) ≥ 0.5 → Attack
```

**Baseline performance (clean test set)**

| Metric | Value |
|:-------|:-----:|
| Precision | 98.50% |
| Recall | 85.46% |
| F1‑Score | **91.52%** |

> ⚠️ Note: 14.5% of attacks are already missed by the vanilla model — a structural blind spot that adversarial attacks will ruthlessly exploit.

---

## 🔴 Red Teaming: Offensive Adversarial ML

We generate adversarial examples $x^{adv}$ that flip the model's prediction from **attack (1) → benign (0)**, while staying physically valid.

### 🔹 Constraint Projection $\mathcal{P}_c$ — The Game Changer

Unlike image attacks, **network features cannot be arbitrarily modified**.
We partition features into two groups:

- **Unilateral** *(can only increase)*: `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `sload`, `dload` — because you cannot reduce past traffic.
- **Free** *(bidirectional)*: derived ratios, statistics.

$$[\mathcal{P}_c(\tilde{x})]_j = \begin{cases} \max(\tilde{x}_j, x_j) & \text{if unilateral} \\ \text{clip}(\tilde{x}_j,\, x_j - \delta_j,\, x_j + \delta_j) & \text{otherwise} \end{cases}$$

> This makes our attacks **operationally realistic** — not just mathematical curiosities.

---

### ⚔️ Attack 1: Constrained Projected Gradient Descent (CPGD)

**White‑box** (full model access, gradients). Iterative FGSM + projection:

$$\mathbf{g}^{(t)} = \nabla_{\mathbf{x}} \mathcal{L}_{\text{BCE}}(f(\mathbf{x}^{(t)}), 1)$$

$$\tilde{\mathbf{x}}^{(t+1)} = \mathbf{x}^{(t)} + \epsilon \cdot \operatorname{sign}(\mathbf{g}^{(t)})$$

$$\mathbf{x}^{(t+1)} = \mathcal{P}_c(\tilde{\mathbf{x}}^{(t+1)})$$

| Attack Success Rate (ASR) | **53.23%** |
|:--------------------------|:----------:|

The model is fooled on more than **one out of two attacks** — a significant vulnerability.

---

### 🐝 Attack 2: Particle Swarm Optimization (PSO)

**Black‑box** (only predictions, no gradients). A swarm of $N$ particles explores the constrained space, moving with inertia and social/cognitive components:

$$\mathbf{v}_{i}^{(t+1)} = \omega\mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i^{(t)})$$

| Parameter | Value |
|:----------|:-----:|
| Swarm size N | 30 |
| Iterations T | 40 |
| Inertia ω | 0.7 |
| Cognitive c₁ | 1.5 |
| Social c₂ | 1.5 |

| Attack Success Rate (ASR) | **97.95%** |
|:--------------------------|:----------:|

The black‑box swarm **almost completely evades** the IDS — a striking paradox: *no gradients → higher success*.

---

### 📊 Comparison

| Attack | Paradigm | Knowledge | ASR | Complexity |
|:-------|:---------|:----------|:----|:-----------|
| CPGD | Gradient | White‑box | 53.23% | $O(T \cdot d)$ |
| PSO | Swarm | Black‑box | **97.95%** | $O(T \cdot N \cdot d)$ |

**Why does black‑box work better?**
- The constrained gradient landscape is non‑convex and has "masked" gradients — local ascent gets stuck.
- PSO's stochastic global search discovers adversarial basins that gradient‑based methods cannot reach.

---

## 🔵 Blue Teaming: Adversarial Training

We apply Madry's **min‑max** formulation:

$$\min_{\theta} \; \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \max_{\mathbf{x}' \in \mathcal{C}(\mathbf{x})} \mathcal{L}(f_{\theta}(\mathbf{x}'), y) \right]$$

Training loop (5‑step fast CPGD per batch):

```python
for epoch in range(E):
    for X_batch, y_batch in dataloader:
        X_adv = cpgd_fast(X_batch, model)   # inner max
        X_total = concat(X_batch, X_adv)
        y_total = concat(y_batch, y_batch)
        loss = BCE_with_logits(model(X_total), y_total)
        loss.backward()
        optimizer.step()
```

> ⚠️ **No data leakage:** adversarial examples are generated only from the training set.

---

## 📈 Results: Before vs After Vaccination

| Phase | Metric | Baseline Model | Adversarially Trained | Delta |
|:------|:-------|:--------------:|:---------------------:|:-----:|
| **Clean classification** | Precision | 98.50% | ≈97.8% | -0.7% |
| | Recall | 85.46% | ≈88.2% | **+2.7%** ✅ |
| | F1‑Score | 91.52% | ≈92.8% | **+1.3%** ✅ |
| **Red Teaming** | ASR – CPGD (white‑box) | 53.23% | **2.41%** | 🛡️ -95.5% |
| | ASR – PSO (black‑box) | 97.95% | **49.61%** | 🛡️ -49.3% |

### 🔥 Key Insights

- ✅ **Robustness without degradation:** Recall even *improves* slightly — adversarial examples act as a constructive data augmentation.
- 🧬 **Cross‑attack transfer:** Training only against CPGD cuts PSO's success rate in half. The model learns more general decision boundaries.
- ⚠️ **Residual risk:** PSO still fools the model in 49.6% of cases → future work must include mixed adversarial training.

---

## 🚀 How to Run (Reproducibility)

**1️⃣ Clone the repository**
```bash
git clone https://github.com/AymanMidan/Adversarial-Machine-Learning-applied-to-IDS
cd Adversarial-Machine-Learning-applied-to-IDS

```

**2️⃣ Install dependencies**
```bash
pip install torch pandas numpy scikit-learn matplotlib tqdm
```

**3️⃣ Download UNSW‑NB15**

Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the [official source](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them in `data/`.

**4️⃣ Train the baseline model**


**5️⃣ Run Red Team attacks**


**6️⃣ Adversarial training (Blue Team)**


**7️⃣ Evaluate robustness**


## 🧩 Project Structure

```
adversarial-ids/
│
├── 📄 Adversarial_IDS.pdf           # Project report
│
├── 🐍 pretraitement.py              # Feature selection, z-score normalisation, 80/20 split
├── 🐍 modele.py                     # MLP architecture (42→64→32→1) + baseline training
│
├── 🐍 attaque_cpgd.py               # White-box attack: Constrained PGD
├── 🐍 attaque_pso.py                # Black-box attack: Particle Swarm Optimization
│
├── 🐍 defense_adv_training.py       # Blue Team: adversarial min-max training loop
│
├── 💾 mon_baseline_ids.pth          # Saved baseline model weights
├── 💾 mon_modele_robuste.pth        # Saved adversarially trained model weights
│
├── 📊 NUSW-NB15_features.csv        # Feature descriptions
├── 📊 UNSW_NB15_training-set.csv    # Training set (~82k samples)
├── 📊 UNSW_NB15_testing-set.csv     # Test set
│
└── 📄 README.md
```

---

## 🧠 Discussion: The White‑Box vs Black‑Box Paradox

> How can a black‑box attack (PSO, **97.95%**) outperform a white‑box gradient attack (CPGD, **53.23%**) on the same model?

**Explanation:** The constrained projection $\mathcal{P}_c$ removes gradient information in unilateral dimensions. This creates **gradient masking** — the remaining gradient points to suboptimal directions. PSO, being derivative‑free, does not suffer from this and can traverse the non‑convex loss landscape more globally.

**Implication:** Never trust white‑box robustness alone. Always include black‑box evaluations (evolutionary, query‑based) to uncover hidden blind spots.

---

## 🔮 Future Work

| Timeframe | Direction |
|:----------|:----------|
| Short‑term | Mixed adversarial training (CPGD + PSO) to drive PSO ASR below 10% |
| Short‑term | Randomized smoothing for certified robustness bounds |
| Mid‑term | Multi‑label IDS (9 attack families) to study per‑family robustness |
| Mid‑term | Adaptive attacker that knows the defense strategy |
| Long‑term | Deployment on real PCAP traffic with online feature extraction |
| Long‑term | Federated Learning setting for distributed IoT intrusion detection |

---

## 📝 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Ayman MIDAN** · **Mohamed EL HARCHALI**

---

## 📚 References

1. Madry et al. (2018) – *Towards Deep Learning Models Resistant to Adversarial Attacks*
2. Kennedy & Eberhart (1995) – *Particle Swarm Optimization*
3. Moustafa & Slay (2015) – *UNSW-NB15 dataset*
4. Goodfellow et al. (2015) – *Explaining and Harnessing Adversarial Examples (FGSM)*

---

## ⭐ If you find this work useful

Please **star this repository** and cite the project:

```bibtex
@misc{midan2025adversarialids,
  author    = {Midan, Ayman and El Harchali, Mohamed},
  title     = {Adversarial Machine Learning Applied to Intrusion Detection Systems},
  year      = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/AymanMidan/Adversarial-Machine-Learning-applied-to-IDS
}}
}
```

---

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  "A model that is excellent on clean data can be trivially      │
│   broken by an informed attacker. Robustness must be earned,    │
│   certified, and maintained iteratively."                       │
│                                                                 │
│                              — Project conclusion               │
└─────────────────────────────────────────────────────────────────┘
```

**🔒 Stay secure. Think adversarial.**

*Deep Learning — 2025-2026*

</div>
