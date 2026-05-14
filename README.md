# 🛡️ Adversarial-IDS: Red Teaming & Blue Teaming on Deep Learning Networks

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Cybersecurity](https://img.shields.io/badge/Domain-Cybersecurity%20%26%20Digital%20Trust-black?logo=security)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Projet d'Ingénierie** : Étude complète du cycle de vie sécuritaire (Build - Break - Fix) d'un Système de Détection d'Intrusion (IDS) face aux attaques adversarielles sous contraintes physiques réseau.

## 📖 Sommaire
- [À propos du projet](#-à-propos-du-projet)
- [L'Innovation : La Contrainte Physique](#-linnovation--la-contrainte-physique)
- [Architecture et Scripts](#-architecture-et-scripts)
- [Résultats et Métriques](#-résultats-et-métriques)
- [Installation et Utilisation](#-installation-et-utilisation)
- [Auteur](#-auteur)

---

## 🎯 À propos du projet

L'intégration du Deep Learning dans les SOC (Security Operations Centers) offre des capacités de détection inédites. Cependant, ces modèles sont vulnérables aux **exemples adversariaux**. 

Ce projet démontre qu'un modèle MLP hautement performant (F1-Score > 91%) sur le dataset de référence **UNSW-NB15** peut être complètement aveuglé par des attaques mathématiques et heuristiques. Plus important encore, ce projet implémente une défense robuste par **Adversarial Training**, capable de restaurer la confiance numérique du système.

### Les 3 phases du projet :
1. **Blue Team (Build)** : Développement d'un IDS Baseline en PyTorch.
2. **Red Team (Break)** : Attaques en Boîte Blanche (CPGD) et Boîte Noire (PSO) pour générer des flux malveillants indétectables.
3. **Blue Team (Fix)** : Entraînement robuste (Vaccination) empêchant l'évasion.

---

## ⚠️ L'Innovation : La Contrainte Physique

Contrairement à la vision par ordinateur où un pixel peut être modifié librement, le trafic réseau obéit à la loi de la causalité. **On ne peut pas réduire la durée d'une connexion passée ni supprimer des paquets déjà envoyés.**

Ce projet implémente une fonction de **Projection Physique** rigoureuse : l'attaquant (l'algorithme) ne peut qu'ajouter du délai ou insérer du *padding*. Les attaques générées sont donc 100% réalistes et exploitables dans le monde réel.

---

## 🏗️ Architecture et Scripts

Le projet est divisé en modules indépendants pour une approche MLOps claire :

```text
📁 Adversarial-IDS/
├── 📄 pretraitement.py         # Nettoyage, normalisation (Z-score) et split du dataset UNSW-NB15
├── 📄 modele.py                # Architecture du MLP (Baseline) et boucle d'entraînement
├── 📄 attaque_cpgd.py          # Attaque Boîte Blanche (Constrained Projected Gradient Descent)
├── 📄 attaque_pso.py           # Attaque Boîte Noire (Particle Swarm Optimization)
├── 📄 defense_adv_training.py  # Entraînement robuste (Mixage intra-boucle sans fuite de données)
└── 📄 api_serveur.py           # (Optionnel) Déploiement FastAPI pour l'inférence en temps réel
