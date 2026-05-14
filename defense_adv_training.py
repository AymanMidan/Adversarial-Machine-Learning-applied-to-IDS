import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from pretraitement import get_preprocessed_data
from modele import IDS_MLP


# =====================================================================
# 1. OUTILS POUR LE "VACCIN" ADVERSARIEL
# =====================================================================
def get_feature_indices():
    df = pd.read_csv('UNSW_NB15_training-set.csv', nrows=0)
    cols = df.drop(columns=['id', 'attack_cat', 'label'], errors='ignore').columns.tolist()
    return {"dur": cols.index("dur"), "sbytes": cols.index("sbytes"), "sinpkt": cols.index("sinpkt")}


def generate_fast_cpgd(model, X_batch, y_batch, indices, alpha=0.1, iters=5):
    """
    Génère une attaque rapide à la volée pendant l'entraînement.
    On utilise moins d'itérations (5) pour ne pas ralentir l'apprentissage.
    """
    X_adv = X_batch.clone().detach().requires_grad_(True)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(iters):
        outputs = model(X_adv)
        loss = criterion(outputs, y_batch)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            # Projection physique
            for idx in indices.values():
                X_adv[:, idx] = torch.max(X_batch[:, idx], X_adv[:, idx])

        X_adv.requires_grad_(True)

    return X_adv.detach()


# =====================================================================
# 2. ENTRAÎNEMENT ROBUSTE (Adversarial Training)
# =====================================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_preprocessed_data()
    indices = get_feature_indices()

    batch_size = 256
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model_robuste = IDS_MLP(input_size=42).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_robuste.parameters(), lr=0.001)

    print("--- Début de l'Entraînement Robuste (Blue Teaming) ---")
    epochs = 10

    for epoch in range(epochs):
        model_robuste.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 1. On isole les malwares du batch actuel pour les déguiser
            mask_attacks = (batch_y == 1).squeeze()

            # Si le batch contient des attaques, on génère le "vaccin"
            if mask_attacks.sum() > 0:
                X_attacks = batch_X[mask_attacks]
                y_attacks = batch_y[mask_attacks]

                # Génération des exemples adversariels (le vaccin)
                X_adv_attacks = generate_fast_cpgd(model_robuste, X_attacks, y_attacks, indices)

                # On combine les données saines avec les données adversarielles
                batch_X_mixed = torch.cat([batch_X, X_adv_attacks], dim=0)
                batch_y_mixed = torch.cat([batch_y, y_attacks], dim=0)
            else:
                batch_X_mixed = batch_X
                batch_y_mixed = batch_y

            # 2. Entraînement classique sur le batch mélangé
            optimizer.zero_grad()
            outputs = model_robuste(batch_X_mixed)
            loss = criterion(outputs, batch_y_mixed)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Époque [{epoch + 1}/{epochs}] - Perte moyenne : {running_loss / len(train_loader):.4f}")

    # =====================================================================
    # 3. SAUVEGARDE DU MODÈLE BLINDÉ
    # =====================================================================
    torch.save(model_robuste.state_dict(), "mon_modele_robuste.pth")
    print("\n🛡️ Modèle vacciné sauvegardé sous 'mon_modele_robuste.pth'")