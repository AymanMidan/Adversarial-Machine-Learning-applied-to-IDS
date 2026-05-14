import pandas as pd
import numpy as np
import random
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder



def get_preprocessed_data():
    def set_seed(seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(42)
    print("Graines aléatoires fixées.")

    # 2. CHARGEMENT DES DONNÉES
    # Assure-toi que les fichiers CSV sont dans le même dossier que ton script
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')

    print(f"Taille initiale - Train: {train_df.shape}, Test: {test_df.shape}")

    # 3. NETTOYAGE
    # On supprime l'ID (qui n'a aucune valeur prédictive) et attack_cat (on se concentre sur la classification binaire 'label')
    cols_to_drop = ['id', 'attack_cat']
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])

    # Séparation des caractéristiques (X) et de la cible (y)
    # Attention : dans certains CSV, la colonne s'appelle 'label' et dans d'autres 'Label'
    target_col = 'label' if 'label' in train_df.columns else 'Label'

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    # 4. ENCODAGE DES VARIABLES CATÉGORIQUES (Texte -> Nombres)
    cat_cols = ['proto', 'service', 'state']

    for col in cat_cols:
        le = LabelEncoder()
        # On ajuste (fit) sur la combinaison de train et test pour éviter les erreurs
        # si une catégorie rare n'existe que dans le set de test
        combined_data = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined_data)

        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # 5. NORMALISATION
    # On sélectionne uniquement les colonnes numériques
    num_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns

    scaler = StandardScaler()
    # On "fit" UNIQUEMENT sur le set d'entraînement pour éviter la fuite de données (data leakage)
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    # On transforme le test set avec les paramètres appris sur le train set
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Conversion finale en Tenseurs PyTorch (Prêt pour le Deep Learning)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    print(f"Tenseurs finaux - X_train: {X_train_tensor.shape}, y_train: {y_train_tensor.shape}")
    print("Prétraitement terminé avec succès !")

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor