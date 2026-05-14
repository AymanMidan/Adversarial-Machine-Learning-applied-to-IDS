<div align="center">

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    ██████╗ ███████╗██████╗     ████████╗███████╗ █████╗      ║
║    ██╔══██╗██╔════╝██╔══██╗       ██╔══╝██╔════╝██╔══██╗     ║
║    ██████╔╝█████╗  ██║  ██║       ██║   █████╗  ███████║     ║
║    ██╔══██╗██╔══╝  ██║  ██║       ██║   ██╔══╝  ██╔══██║     ║
║    ██║  ██║███████╗██████╔╝       ██║   ███████╗██║  ██║     ║
║    ╚═╝  ╚═╝╚══════╝╚═════╝        ╚═╝   ╚══════╝╚═╝  ╚═╝     ║
║                                                               ║
║          ⚔️  ADVERSARIAL  IDS  ⚔️                             ║
╚═══════════════════════════════════════════════════════════════╝
```

# Adversarial Machine Learning appliqué aux IDS
### Red Teaming & Blue Teaming d'un réseau de neurones MLP sous PyTorch

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-00C896?style=for-the-badge)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![INPT](https://img.shields.io/badge/INPT-2025--2026-8B0000?style=for-the-badge)](https://www.inpt.ac.ma)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

> *"Un modèle excellent sur données propres peut être trivialement compromis par un attaquant informé.  
> La robustesse s'acquiert, se certifie et se maintient de manière itérative."*

</div>

---

## 🎯 Vue d'Ensemble

Ce projet explore la **vulnérabilité et la robustification** d'un Système de Détection d'Intrusion (IDS) basé sur le Deep Learning face aux **attaques adversariales réalistes**. L'originalité centrale : nos attaques respectent les **contraintes physiques du trafic réseau** — elles ne modifient que ce qu'un vrai attaquant pourrait modifier.

### Le Cycle Complet : Build → Break → Fix

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   🏗️  BUILD     │────▶│   🔴 RED TEAM   │────▶│   🔵 BLUE TEAM  │
│                 │     │                 │     │                 │
│  IDS Baseline   │     │  CPGD  │  PSO   │     │  Adversarial    │
│  F1 = 91.52%    │     │ 53.23% │ 97.95% │     │  Training       │
│                 │     │  ASR   │  ASR   │     │  CPGD → 2.41%   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 📊 Résultats en un Coup d'Œil

<div align="center">

| Phase | Métrique | Modèle Naïf | Modèle Vacciné | Delta |
|:------|:---------|:-----------:|:--------------:|:-----:|
| **Classification** | Précision | 98.50% | ~97.8% | -0.7% |
| | Rappel | 85.46% | ~88.2% | **+2.7%** ✅ |
| | F1-Score | 91.52% | ~92.8% | **+1.3%** ✅ |
| **Red Team** | ASR – CPGD (Boîte Blanche) | 53.23% | **2.41%** | 🛡️ -95.5% |
| | ASR – PSO (Boîte Noire) | 97.95% | **49.61%** | 🛡️ -49.3% |

</div>

> **Observation clé :** La vaccination adversariale *améliore* légèrement le rappel sur données propres — les exemples adversariaux contraints enrichissent le jeu d'entraînement de manière constructive.

---

## 🧠 Architecture

### Le Modèle MLP Baseline

```python
Input (d=42 features)
        │
        ▼
┌───────────────────┐
│  Linear(42 → 64)  │
│  ReLU             │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Linear(64 → 32)  │
│  ReLU             │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Linear(32 → 1)   │  ──▶  σ(ŷ) ≥ 0.5  →  Attaque
│  (logit)          │  ──▶  σ(ŷ) < 0.5  →  Bénin
└───────────────────┘
```

**Perte :** Binary Cross-Entropy with Logits  
**Optimiseur :** Adam (lr=1e-3, weight decay=1e-4)  
**Entraînement :** 20 époques, batch size 256

---

## 📁 Structure du Projet

```
adversarial-ids/
│
├── 📂 data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
│
├── 📂 src/
│   ├── 🔧 preprocessing.py        # Pipeline z-score, split stratifié
│   ├── 🏗️  model.py                # Architecture MLP PyTorch
│   ├── 🔴 attacks/
│   │   ├── cpgd.py               # Constrained PGD (boîte blanche)
│   │   └── pso.py                # Particle Swarm Optimization (boîte noire)
│   └── 🔵 defense/
│       └── adversarial_training.py  # Boucle min-max de Madry
│
├── 📂 notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_red_teaming.ipynb
│   └── 03_blue_teaming.ipynb
│
├── 📂 results/
│   └── metrics_summary.json
│
├── 📄 rapport.pdf
└── 📄 README.md
```

---

## ⚔️ Red Teaming

### Attaque 1 — CPGD : Boîte Blanche

Le **Constrained Projected Gradient Descent** adapte l'attaque PGD de Madry en intégrant une projection physique à chaque itération.

**L'intuition :** Un attaquant peut ajouter du padding à ses paquets ou injecter du délai — mais il ne peut pas voyager dans le temps pour réduire la durée d'une session déjà terminée.

```python
for t in range(T):
    g = ∇_x L_BCE(f(x_t), y=1)      # Gradient ascendant
    x_tilde = x_t + ε · sign(g)      # Pas FGSM
    x_{t+1} = P_C(x_tilde)           # Projection physique ← LA CLÉ
```

**Projection physique P_C :**
```
[P_C(x̃)]_j = max(x̃_j, x_j)          si feature unilatérale (durée, octets...)
             clip(x̃_j, x_j-δ, x_j+δ)  sinon (ratios, statistiques dérivées)
```

```
📊 Résultat CPGD : ASR = 53.23%
   L'IDS est trompé sur plus d'une attaque sur deux.
```

---

### Attaque 2 — PSO : Boîte Noire

La **Particle Swarm Optimization** n'a besoin d'aucun gradient — uniquement de la sortie de l'IDS (0 ou 1). Chaque particule est un exemple adversarial candidat, l'essaim converge vers les angles morts du modèle.

```
Essaim de N=30 particules, T=40 itérations

v_{i}^{t+1} = ω·v_i^t  +  c₁r₁(pBest_i - x_i^t)  +  c₂r₂(gBest - x_i^t)
                              ↑                              ↑
                       mémoire individuelle          mémoire collective
```

| Paramètre | Valeur |
|:----------|:------:|
| Taille essaim N | 30 |
| Itérations T | 40 |
| Inertie ω | 0.7 |
| Cognitif c₁ | 1.5 |
| Social c₂ | 1.5 |

```
🚨 Résultat PSO : ASR = 97.95%
   Près de 98 attaques sur 100 passent inaperçues.
```

---

### ⚠️ Le Paradoxe Boîte Blanche / Boîte Noire

> La boîte noire (PSO, 97.95%) est **presque deux fois plus efficace** que la boîte blanche (CPGD, 53.23%).

**Explication :** Le CPGD suit le gradient local qui, dans l'espace contraint, peut être partiellement annulé par la projection P_C — un phénomène de **gradient masking**. Le PSO explore l'espace de manière stochastique et découvre des bassins adversariaux inaccessibles au gradient. Ce résultat contre-intuitif est un signal d'alarme : **l'attaque la plus accessible est la plus dévastatrice**.

---

## 🛡️ Blue Teaming

### Entraînement Adversarial (Paradigme de Madry)

L'objectif standard est remplacé par un problème **min-max** :

```
min_θ E_(x,y)~D [ max_{x'∈C(x)} L(f_θ(x'), y) ]
                   ↑
           Le problème interne génère les exemples les plus difficiles
```

**Implémentation :** CPGD rapide (K=5 itérations) intra-boucle d'entraînement.

```python
for epoch in range(E):
    for X_batch, y_batch in dataloader:
        # 1. Générer les exemples adversariaux
        X_adv = cpgd_fast(X_batch, model, steps=5)
        
        # 2. Concaténer propres + adversariaux
        X_aug = concat(X_batch, X_adv)
        y_aug = concat(y_batch, y_batch)  # labels inchangés
        
        # 3. Mise à jour du modèle
        loss = BCE(model(X_aug), y_aug)
        optimizer.step(loss)
```

> **Garantie d'intégrité :** Les exemples adversariaux sont générés **uniquement sur le Training Set**. Le Test Set n'est jamais exposé pendant l'entraînement.

### Résultats Post-Vaccination

```
CPGD  :  53.23%  ──────────────────────▶  2.41%   (-95.5%) 🛡️🛡️🛡️
PSO   :  97.95%  ──────────────▶  49.61%           (-49.3%) 🛡️🛡️
```

La **robustesse croisée** au PSO (jamais vu pendant la défense) s'explique par le fait que l'entraînement adversarial CPGD force le modèle à apprendre des représentations plus généralisables — la frontière de décision robuste aux perturbations de gradient l'est aussi aux perturbations évolutionnaires.

---

## 🚀 Installation & Usage

```bash
# Cloner le repo
git clone https://github.com/<your-username>/adversarial-ids.git
cd adversarial-ids

# Environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Dépendances
pip install -r requirements.txt
```

### Reproduire les expériences

```bash
# 1. Entraîner le modèle baseline
python src/train_baseline.py

# 2. Red Teaming
python src/attacks/cpgd.py --steps 20 --epsilon 0.1
python src/attacks/pso.py --n-particles 30 --iterations 40

# 3. Blue Teaming
python src/defense/adversarial_training.py --epochs 20

# 4. Évaluation complète
python src/evaluate.py --model vacciné
```

### Dépendances principales

```
torch >= 2.0
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
matplotlib >= 3.7
```

---

## 📐 Dataset : UNSW-NB15

| Attribut | Valeur |
|:---------|:-------|
| Source | Cyber Range Lab, UNSW Sydney (2015) |
| Instances | ~82 500 |
| Features brutes | 49 → **42 sélectionnées** |
| Classes | Bénin (0) / Attaque (1) |
| Équilibre | 51% bénin / 49% attaque |
| Familles d'attaques | DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms, Backdoor, Analysis |

**Pipeline de pré-traitement :**
1. Suppression des colonnes non-numériques (`proto`, `service`, `state`)
2. Normalisation z-score calculée **uniquement sur le train set**
3. Split stratifié 80% / 20%

---

## 📈 Métriques

**Attack Success Rate (ASR) :**

```
ASR = |{x ∈ D_atk : f(x)=1 ∧ f(x_adv)=0}|
      ─────────────────────────────────────
              |{x ∈ D_atk : f(x)=1}|

ASR = 100% → L'IDS ne détecte plus aucune attaque
ASR = 0%   → L'IDS résiste parfaitement
```

---

## 🔬 Limites & Perspectives

### Limites identifiées

- **Dérive temporelle :** UNSW-NB15 date de 2015 ; les attaques modernes (exfiltration DNS/HTTPS, canaux cachés) peuvent ne pas être représentées.
- **Features catégorielles exclues :** `proto`, `service`, `state` pourraient enrichir la topologie adversariale.
- **PSO non intégré à la défense :** ASR résiduel de 49.61% — le risque reste substantiel.
- **Attaquant unique :** Les attaques coordonnées (botnets) ou adaptatives ne sont pas modélisées.

### Perspectives de recherche

**Court terme**
- [ ] Entraînement adversarial mixte (CPGD + PSO) pour combler les 49.61% résiduels
- [ ] Randomized Smoothing pour une certification probabiliste de l'ASR

**Moyen terme**
- [ ] Extension multi-étiquettes sur les 9 familles d'attaques UNSW-NB15
- [ ] Attaquants adaptatifs (connaissant la stratégie de défense)

**Long terme**
- [ ] Déploiement sur trafic réel (PCAP) avec extraction automatique de features
- [ ] Adversarial ML en Federated Learning pour IDS distribués IoT

---

## 👥 Auteurs

<div align="center">

| | |
|:---:|:---:|
| **Ayman MIDAN** | **Mohamed EL HARCHALI** |
| [@ayman-midan](https://github.com/) | [@m-elharchali](https://github.com/) |

**Encadrant :** Pr. Tarik FISSAA  
**Module :** Deep Learning  
**Institution :** Institut National des Postes et Télécommunications (INPT)  
**Année :** 2025–2026

</div>

---

## 📚 Références

1. Madry, A. et al. — *Towards Deep Learning Models Resistant to Adversarial Attacks* — ICLR 2018
2. Kennedy, J. & Eberhart, R. — *Particle Swarm Optimization* — IEEE ICNN 1995
3. Moustafa, N. & Slay, J. — *UNSW-NB15: a comprehensive data set for network intrusion detection systems* — MilCIS 2015

---

<div align="center">

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   La cybersécurité basée sur l'IA ne peut atteindre    │
│   sa maturité que si ses praticiens intègrent           │
│   l'Adversarial ML comme un standard d'audit —         │
│   au même titre que les tests de pénétration.          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

*INPT — Deep Learning — 2025-2026*

</div>
