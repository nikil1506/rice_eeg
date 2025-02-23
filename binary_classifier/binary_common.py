import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA

important_columns = [
    # Schizophrenia (alpha PSD)
    "AB.C.alpha.a.FP1",
    "AB.C.alpha.b.FP2",
    "AB.C.alpha.c.F7",
    "AB.C.alpha.d.F3",
    "AB.C.alpha.e.Fz",
    "AB.C.alpha.f.F4",
    "AB.C.alpha.g.F8",
    "AB.C.alpha.h.T3",
    "AB.C.alpha.i.C3",
    "AB.C.alpha.j.Cz",
    "AB.C.alpha.k.C4",
    "AB.C.alpha.l.T4",
    "AB.C.alpha.m.T5",
    "AB.C.alpha.n.P3",
    "AB.C.alpha.o.Pz",
    "AB.C.alpha.p.P4",
    "AB.C.alpha.q.T6",
    "AB.C.alpha.r.O1",
    "AB.C.alpha.s.O2",
    # Trauma and stress-related disorders (beta FC)
    "COH.D.beta.a.FP1.b.FP2",
    "COH.D.beta.c.F7.d.F3",
    "COH.D.beta.e.Fz.f.F4",
    "COH.D.beta.g.F8.h.T3",
    "COH.D.beta.i.C3.j.Cz",
    "COH.D.beta.k.C4.l.T4",
    "COH.D.beta.m.T5.n.P3",
    "COH.D.beta.o.Pz.p.P4",
    "COH.D.beta.q.T6.r.O1",
    # Anxiety disorders (whole band PSD: all frequency bands)
    "AB.A.delta.a.FP1",
    "AB.A.delta.b.FP2",
    "AB.B.theta.a.FP1",
    "AB.B.theta.b.FP2",
    # "AB.C.alpha.a.FP1",
    # "AB.C.alpha.b.FP2",
    "AB.D.beta.a.FP1",
    "AB.D.beta.b.FP2",
    "AB.E.highbeta.a.FP1",
    "AB.E.highbeta.b.FP2",
    "AB.F.gamma.a.FP1",
    "AB.F.gamma.b.FP2",
    # Mood disorders (theta FC)
    "COH.B.theta.a.FP1.b.FP2",
    "COH.B.theta.c.F7.d.F3",
    "COH.B.theta.e.Fz.f.F4",
    "COH.B.theta.g.F8.h.T3",
    "COH.B.theta.i.C3.j.Cz",
    "COH.B.theta.k.C4.l.T4",
    "COH.B.theta.m.T5.n.P3",
    "COH.B.theta.o.Pz.p.P4",
    "COH.B.theta.q.T6.r.O1",
    # Addictive disorders (theta PSD)
    # "AB.B.theta.a.FP1",
    # "AB.B.theta.b.FP2",
    "AB.B.theta.c.F7",
    "AB.B.theta.d.F3",
    "AB.B.theta.e.Fz",
    "AB.B.theta.f.F4",
    "AB.B.theta.g.F8",
    "AB.B.theta.h.T3",
    "AB.B.theta.i.C3",
    "AB.B.theta.j.Cz",
    "AB.B.theta.k.C4",
    "AB.B.theta.l.T4",
    "AB.B.theta.m.T5",
    "AB.B.theta.n.P3",
    "AB.B.theta.o.Pz",
    "AB.B.theta.p.P4",
    "AB.B.theta.q.T6",
    "AB.B.theta.r.O1",
    "AB.B.theta.s.O2",
    # Obsessive-compulsive disorder (gamma FC)
    "COH.F.gamma.a.FP1.b.FP2",
    "COH.F.gamma.c.F7.d.F3",
    "COH.F.gamma.e.Fz.f.F4",
    "COH.F.gamma.g.F8.h.T3",
    "COH.F.gamma.i.C3.j.Cz",
    "COH.F.gamma.k.C4.l.T4",
    "COH.F.gamma.m.T5.n.P3",
    "COH.F.gamma.o.Pz.p.P4",
    "COH.F.gamma.q.T6.r.O1",
]


def read_file(file_path, columns=None):
    if columns is None:
        data = pd.read_csv(file_path).drop(
            columns=[
                "ID",
                "eeg.date",
                "main.disorder",
                "specific.disorder",
                "age",
                "IQ",
            ]
        )
    else:
        data = pd.read_csv(file_path, usecols=columns)
    if "sex" in data.columns:
        data["sex"] = data["sex"].map({"M": 0, "F": 1})
    data = data.dropna(how="all", axis=1)
    data = data.fillna(data.mean())
    weight_factor = 2.0
    data[important_columns] *= weight_factor
    return data


relevant_columns = read_file("data/train.csv").columns.tolist()
false_label = "Healthy control"


def binary_training(true_label):
    # Load dataset with only relevant columns
    data = read_file("data/train.csv", relevant_columns)

    # Features and labels
    X = data.values
    y = (
        pd.read_csv("data/train.csv")["main.disorder"]
        .apply(lambda x: {false_label: 0, true_label: 1}.get(x, 0))
        .values.astype(int)
    )

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler

    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(X_scaled)

    joblib.dump(scaler, f"{true_label}_Scaler.pkl")
    joblib.dump(pca, f"{true_label}_Pca.pkl")

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    # ---------------- Hyperparameter Tuning (Grid Search) ----------------
    param_grid = {
        "C": [0.1, 0.5, 1, 5, 10],  # Regularization parameter
        "kernel": ["linear", "rbf"],  # Linear and RBF kernel
    }

    svm = SVC(probability=True, class_weight="balanced", random_state=42)

    grid_search = GridSearchCV(
        svm, param_grid, cv=10, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_

    # Save the trained SVM model
    joblib.dump(best_svm, f"{true_label}_Tuned.pkl")
    # ---------------- Evaluate Best Model ----------------
    y_pred = best_svm.predict(X_test)
    y_pred_prob = best_svm.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n Tuned SVM Model for {true_label} vs. {false_label}")
    print(f" Best Parameters: {grid_search.best_params_}")
    print(f" Accuracy: {accuracy:.4f}")
    print(f" AUC Score: {auc:.4f}")
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Find the optimal threshold (Youden’s J statistic: max(TPR - FPR))
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"🔹 Optimal Decision Threshold: {optimal_threshold:.4f}")

    # Convert probabilities to labels using the optimal threshold
    pred_labels = (y_pred_prob >= optimal_threshold).astype(int)
    print(pred_labels)
    return optimal_threshold


def predict_mood_disorder(new_data_path, true_label):
    """
    Loads new EEG data, scales it, and predicts Mood Disorder probability.
    """
    # Load saved scaler and model
    scaler = joblib.load(f"{true_label}_Scaler.pkl")
    svm = joblib.load(f"{true_label}_Tuned.pkl")
    pca = joblib.load(f"{true_label}_Pca.pkl")

    # Load new data
    cols = relevant_columns.copy()
    new_data = read_file(new_data_path, cols)

    # Convert DataFrame to NumPy before applying StandardScaler
    new_data_scaled = scaler.fit_transform(new_data.values)
    new_data_pca = pca.fit_transform(new_data_scaled)

    # Predict probabilities
    predictions = svm.predict_proba(new_data_pca)[:, 1]

    return predictions


def decode_predictions(preds, true_label):
    """
    Convert predictions (0,1) back to disorder labels.
    """
    label_map = {0: false_label, 1: true_label}
    return [label_map[pred] for pred in preds]


def get_prediction_label(pred, true_label, optimal_threshold):
    value = (pred >= optimal_threshold).astype(int)
    label_map = {0: false_label, 1: true_label}
    return label_map[value]


def binary_predictions(optimal_threshold, true_label):
    preds = predict_mood_disorder("data/test.csv", true_label)
    # Example usage
    pred_labels = (preds >= optimal_threshold).astype(
        int
    )  # Convert probabilities to binary labels
    decoded_labels = decode_predictions(pred_labels, true_label)

    print("\nPredicted Labels:\n", decoded_labels)
    return preds


def get_binary_accuracy(df, true_label):
    correct_predictions = (
        (df["y_pred"] == true_label) & (df["y_true"] == true_label)
    ) | ((df["y_true"] != true_label) & (df["y_pred"] != true_label))
    accuracy = correct_predictions.sum() / len(df)
    print(f"Accuracy: {accuracy:.2f}")


def get_accuracy(df):
    accuracy = (df["y_true"] == df["y_pred"]).mean()
    print(f"Accuracy: {accuracy:.2f}")
