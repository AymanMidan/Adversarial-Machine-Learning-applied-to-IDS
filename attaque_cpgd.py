import torch
import torch.nn as nn
import pandas as pd
from pretraitement import get_preprocessed_data
from modele import IDS_MLP  # On importe juste la classe, pas le script entier


# =====================================================================
# 1. RÉCUPÉRATION DYNAMIQUE DES INDICES (La rigueur)
# =====================================================================
def get_feature_indices():
    # On charge juste les noms de colonnes pour trouver les index
    df = pd.read_csv('UNSW_NB15_training-set.csv', nrows=0)
    # On applique le même nettoyage que dans ton pretraitement.py
    cols = df.drop(columns=['id', 'attack_cat', 'label'], errors='ignore').columns.tolist()

    # On cherche les index exacts par nom
    indices = {
        "dur": cols.index("dur"),
        "sbytes": cols.index("sbytes"),
        "sinpkt": cols.index("sinpkt")
    }
    return indices


# =====================================================================
# 2. L'ATTAQUE CPGD AVEC PROJECTION PHYSIQUE
# =====================================================================
class CPGDAttack:
    def __init__(self, model, indices):
        self.model = model
        self.indices = indices  # Dictionnaire des index {nom: index}

    def project(self, X_adv, X_original):
        X_projected = X_original.clone()

        # Pour chaque colonne manipulable, on applique la contrainte 'Uniquement Augmenter'
        for name, idx in self.indices.items():
            # Loi physique : l'attaquant ne peut que rajouter du délai ou du padding
            X_projected[:, idx] = torch.max(X_original[:, idx], X_adv[:, idx])

        return X_projected

    def generate(self, X_batch, y_batch, alpha=0.01, iters=40):
        X_adv = X_batch.clone().detach().requires_grad_(True)
        criterion = nn.BCEWithLogitsLoss()

        for i in range(iters):
            outputs = self.model(X_adv)
            loss = criterion(outputs, y_batch)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                # Pas de gradient (FGSM)
                X_adv = X_adv + alpha * X_adv.grad.sign()
                # PROJECTION : On force le respect des lois du réseau
                X_adv = self.project(X_adv, X_batch)

            X_adv.requires_grad_(True)

        return X_adv


# =====================================================================
# 3. SCRIPT D'EXÉCUTION (Dans attaque_cpgd.py)
# =====================================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_preprocessed_data()

    device = torch.device("cpu")
    model = IDS_MLP(input_size=42).to(device)

    # -- LA LIGNE MAGIQUE : On charge les paramètres appris --
    model.load_state_dict(torch.load('mon_modele_robuste.pth'))

    # On identifie les index
    idx_map = get_feature_indices()
    print(f"Indices identifiés dynamiquement : {idx_map}")

    # On filtre les vraies attaques
    model.eval()  # OBLIGATOIRE avant une attaque
    mask = (y_test == 1).squeeze()
    X_malicious = X_test[mask]
    y_malicious = y_test[mask]

    # Lancement de l'attaque
    attacker = CPGDAttack(model, idx_map)
    # Augmente un peu l'alpha et les itérations pour une attaque plus agressive
    X_adv = attacker.generate(X_malicious, y_malicious, alpha=0.1, iters=50)

    # Évaluation
    with torch.no_grad():
        preds_before = (torch.sigmoid(model(X_malicious)) >= 0.5).float()
        preds_after = (torch.sigmoid(model(X_adv)) >= 0.5).float()

        detected_before = preds_before.sum().item()
        detected_after = preds_after.sum().item()

        evaded = detected_before - detected_after
        asr = (evaded / detected_before) * 100 if detected_before > 0 else 0

        print(f"Attaques détectées AVANT CPGD : {int(detected_before)}/{len(X_malicious)}")
        print(f"Attaques détectées APRÈS CPGD : {int(detected_after)}/{len(X_malicious)}")
        print(f"🔥 Attack Success Rate (ASR) : {asr:.2f}%")