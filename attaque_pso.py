import torch
import torch.nn as nn
import pandas as pd
from pretraitement import get_preprocessed_data
from modele import IDS_MLP


# =====================================================================
# 1. RÉCUPÉRATION DES INDICES
# =====================================================================
def get_feature_indices():
    df = pd.read_csv('UNSW_NB15_training-set.csv', nrows=0)
    cols = df.drop(columns=['id', 'attack_cat', 'label'], errors='ignore').columns.tolist()
    return {"dur": cols.index("dur"), "sbytes": cols.index("sbytes"), "sinpkt": cols.index("sinpkt")}


# =====================================================================
# 2. L'ATTAQUE PSO (BLACK-BOX)
# =====================================================================
class PSOAttack:
    def __init__(self, model, indices, swarm_size=30, max_iters=40):
        self.model = model
        self.indices = indices
        self.swarm_size = swarm_size
        self.max_iters = max_iters

        # Hyperparamètres fixes
        self.c1 = 2.0  # Confiance cognitive
        self.c2 = 2.0  # Influence sociale

    def project(self, X_adv, X_original):
        """Applique les contraintes physiques (Impossible de réduire la taille/durée)"""
        X_projected = X_adv.clone()
        for idx in self.indices.values():
            # On utilise unsqueeze pour s'adapter à la dimension de l'essaim
            X_projected[..., idx] = torch.max(X_original[..., idx], X_adv[..., idx])
        return X_projected

    def generate(self, X_batch):
        N, num_features = X_batch.shape
        device = X_batch.device

        # 1. INITIALISATION DE L'ESSAIM
        # On clone chaque malware en 30 exemplaires (les 30 particules)
        X_swarm = X_batch.unsqueeze(0).repeat(self.swarm_size, 1, 1)

        # On injecte un petit chaos de départ (exploration initiale)
        noise = torch.zeros_like(X_swarm)
        for idx in self.indices.values():
            noise[..., idx] = torch.rand(self.swarm_size, N, device=device) * 2.0

        X_swarm = X_swarm + noise
        X_swarm = self.project(X_swarm, X_batch.unsqueeze(0))

        # Vitesse initiale à 0
        V_swarm = torch.zeros_like(X_swarm)

        # 2. MÉMOIRE DE L'ESSAIM
        pBest = X_swarm.clone()
        pBest_scores = torch.ones(self.swarm_size, N, device=device) * float('inf')

        gBest = X_batch.clone()
        gBest_scores = torch.ones(N, device=device) * float('inf')

        # 3. LA DANSE DE L'ESSAIM (Boucle d'optimisation)
        for iteration in range(self.max_iters):

            # L'astuce d'ingénieur : w baisse de 0.9 à 0.4 progressivement
            w = 0.9 - (0.5 * (iteration / self.max_iters))

            # A. ÉVALUATION (Le maître du jeu)
            with torch.no_grad():
                for s in range(self.swarm_size):
                    # On regarde le score de l'IDS. L'objectif est de le MINIMISER (le rendre inoffensif)
                    scores = self.model(X_swarm[s]).squeeze()

                    # Est-ce le record personnel de cette particule ?
                    better_mask = scores < pBest_scores[s]
                    pBest_scores[s][better_mask] = scores[better_mask]
                    pBest[s][better_mask] = X_swarm[s][better_mask]

                    # Est-ce le record absolu de TOUT l'essaim ?
                    global_better_mask = scores < gBest_scores
                    gBest_scores[global_better_mask] = scores[global_better_mask]
                    gBest[global_better_mask] = X_swarm[s][global_better_mask]

            # B. LE MOUVEMENT
            for s in range(self.swarm_size):
                # r1 et r2 : Le chaos aléatoire à chaque étape
                r1 = torch.rand(N, num_features, device=device)
                r2 = torch.rand(N, num_features, device=device)

                # LA FORMULE MAGIQUE DU PSO
                V_swarm[s] = (w * V_swarm[s]) + \
                             (self.c1 * r1 * (pBest[s] - X_swarm[s])) + \
                             (self.c2 * r2 * (gBest - X_swarm[s]))

                # Mise à jour de la position
                X_swarm[s] = X_swarm[s] + V_swarm[s]

            # C. LA PROJECTION (Le mur de la réalité)
            X_swarm = self.project(X_swarm, X_batch.unsqueeze(0))

        # À la fin, on retourne le trophée global : les meilleurs malwares trouvés
        return gBest


# =====================================================================
# 3. EXÉCUTION
# =====================================================================
if __name__ == "__main__":
    print("--- Préparation des données ---")
    _, _, X_test, y_test = get_preprocessed_data()

    device = torch.device("cpu")
    model = IDS_MLP(input_size=42).to(device)

    try:
        model.load_state_dict(torch.load('mon_modele_robuste.pth'))
        model.eval()
        print("\n🚀 Modèle entraîné chargé (Boîte Noire). Prêt pour l'attaque PSO.")
    except FileNotFoundError:
        print("❌ Erreur : Lance d'abord modele.py pour générer le fichier .pth !")
        exit()

    idx_map = get_feature_indices()

    # On isole les attaques
    mask = (y_test == 1).squeeze()
    X_malicious = X_test[mask]
    y_malicious = y_test[mask]

    # Lancement du Red Teaming (Peut prendre 1-2 minutes car il y a 30 particules !)
    print("\nLancement de l'essaim PSO (30 particules) en cours...")
    attacker = PSOAttack(model, idx_map, swarm_size=30, max_iters=40)
    X_adv_pso = attacker.generate(X_malicious)

    with torch.no_grad():
        preds_before = (torch.sigmoid(model(X_malicious)) >= 0.5).float()
        preds_after = (torch.sigmoid(model(X_adv_pso)) >= 0.5).float()

        detected_before = preds_before.sum().item()
        detected_after = preds_after.sum().item()

        evaded = detected_before - detected_after
        asr = (evaded / detected_before) * 100 if detected_before > 0 else 0

        print(f"\n--- Résultat de l'attaque Black-Box (PSO) ---")
        print(f"Attaques détectées AVANT : {int(detected_before)}/{len(X_malicious)}")
        print(f"Attaques détectées APRÈS : {int(detected_after)}/{len(X_malicious)}")
        print(f"🔥 Attack Success Rate (ASR) : {asr:.2f}%")