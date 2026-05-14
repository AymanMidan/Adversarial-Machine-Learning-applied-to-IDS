import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from pretraitement import get_preprocessed_data

# =====================================================================
# 1. ARCHITECTURE DU MODÈLE
# =====================================================================
class IDS_MLP(nn.Module):
    def __init__(self, input_size=42):
        super(IDS_MLP, self).__init__()
        # Architecture compacte pour respecter les contraintes de calcul
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 1 seul neurone de sortie pour la classification binaire
        )

    def forward(self, x):
        # Retourne les logits bruts (BCEWithLogitsLoss appliquera la fonction Sigmoïde de façon plus stable)
        return self.network(x)

if __name__ == "__main__":
    # =====================================================================
    # 2. PRÉPARATION DES DONNÉES (Batching)
    # =====================================================================
    # On suppose que X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    # proviennent de ton script de prétraitement précédent.

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_preprocessed_data()

    batch_size = 256
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # =====================================================================
    # 3. INITIALISATION (Modèle, Perte, Optimiseur)
    # =====================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    model = IDS_MLP(input_size=42).to(device)

    # BCEWithLogitsLoss combine une couche Sigmoid et la perte Binary Cross Entropy
    # C'est la fonction mathématique idéale pour la classification binaire en Deep Learning
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # =====================================================================
    # 4. BOUCLE D'ENTRAÎNEMENT (Training Loop)
    # =====================================================================
    epochs = 10  # 10 itérations suffisent généralement pour converger avec ce dataset

    print("--- Début de l'entraînement ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 1. Remise à zéro des gradients
            optimizer.zero_grad()

            # 2. Propagation avant (Forward pass)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 3. Rétropropagation (Backward pass)
            loss.backward()

            # 4. Mise à jour des poids
            optimizer.step()

            running_loss += loss.item()

        print(f"Époque [{epoch + 1}/{epochs}] - Perte moyenne : {running_loss / len(train_loader):.4f}")

    # =====================================================================
    # 5. ÉVALUATION ET MÉTRIQUES
    # =====================================================================
    model.eval()  # Passage en mode évaluation (désactive certains comportements comme le Dropout s'il y en avait)
    all_preds = []
    all_targets = []

    with torch.no_grad():  # On ne calcule pas les gradients pour économiser la mémoire
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)

            # Obtention des probabilités avec la fonction Sigmoïde
            logits = model(batch_X)
            probs = torch.sigmoid(logits)

            # On convertit les probabilités en prédictions binaires (seuil de 0.5)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    # Calcul des métriques requises par ton protocole d'évaluation
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    print("\n--- Résultats de l'évaluation du Baseline ---")
    print(f"Précision (Precision) : {precision:.4f}")
    print(f"Rappel (Recall)       : {recall:.4f}")
    print(f"F1-Score              : {f1:.4f}")

    # =====================================================================
    # Sauvegarde du modèle
    # =====================================================================
    torch.save(model.state_dict(), "mon_baseline_ids.pth")
    print("✅ Poids du modèle sauvegardés sous 'mon_baseline_ids.pth'")

