# src/train_model.py
# ─────────────────────────────────────────────────────────────
# Load dataset → train classifier → evaluate → save model.
# Run ONCE after data collection. Re-run anytime you add data.
#
# Run: python src/train_model.py
# ─────────────────────────────────────────────────────────────

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble         import RandomForestClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.preprocessing    import LabelEncoder
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics          import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_CSV, MODEL_FILE, ENCODER_FILE, CM_IMAGE,
    MODEL_DIR, OUTPUT_DIR,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS,
    RF_N_ESTIMATORS, RF_MIN_SAMPLES_LEAF,
    MLP_HIDDEN_LAYERS, MLP_MAX_ITER,
    GESTURES
)


def load_and_validate(csv_path: str):
    """
    Load CSV and run sanity checks before training.
    Catches data problems early — before they cause
    silent training failures.
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"\n── Data overview ─────────────────────────")
    print(f"Total samples : {len(df)}")
    print(f"Feature cols  : {len(df.columns) - 1}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts().to_string())

    # Check 1: all expected gestures present
    missing = set(GESTURES) - set(df['label'].unique())
    if missing:
        raise ValueError(f"Missing gesture classes in dataset: {missing}")

    # Check 2: class balance — warn if any class is <80% of median
    counts = df['label'].value_counts()
    median = counts.median()
    for cls, cnt in counts.items():
        if cnt < median * 0.8:
            print(f"\nWARNING: Class '{cls}' has only {cnt} samples "
                  f"(median={median:.0f}). Consider collecting more.")

    # Check 3: no NaN values
    if df.isnull().any().any():
        raise ValueError("Dataset contains NaN values. Re-check collect_data.py.")

    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    return X, y

    # With 26,000 rows, report expected training time
    print(f"\nDataset size: {len(X):,} samples")
    if len(X) > 10000:
        print("Large dataset detected.")
        print("RF training estimate: 2–5 minutes")
        print("Cross-validation (5-fold) estimate: 10–20 minutes")
        print("Tip: if CV is too slow, reduce CV_FOLDS to 3 in config.py")

def train_and_evaluate(X, y):
    """
    Train Random Forest (primary) and MLP (benchmark).
    Evaluate with stratified k-fold CV + held-out test set.
    Return the best model and fitted label encoder.
    """
    # Encode string labels → integers
    # LabelEncoder sorts alphabetically:
    # forward=0, left=1, reverse=2, right=3, stop=4
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nLabel encoding:")
    for cls, idx in zip(le.classes_, range(len(le.classes_))):
        print(f"  {idx} → {cls}")

    # Stratified split: preserves class ratios in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y_encoded
    )
    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}")

    # ── Define models ────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators      = RF_N_ESTIMATORS,
        min_samples_leaf  = RF_MIN_SAMPLES_LEAF,
        max_features      = 'sqrt',   # each split sees sqrt(63)≈8 features
        random_state      = RANDOM_STATE,
        n_jobs            = -1        # use all CPU cores
    )

    mlp = MLPClassifier(
        hidden_layer_sizes = MLP_HIDDEN_LAYERS,
        activation         = 'relu',
        max_iter           = MLP_MAX_ITER,
        random_state       = RANDOM_STATE
    )

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Cross-validation ─────────────────────────────────────
    print(f"\n── {CV_FOLDS}-fold stratified cross-validation ─────────")

    rf_scores  = cross_val_score(rf,  X, y_encoded, cv=skf, scoring='accuracy')
    mlp_scores = cross_val_score(mlp, X, y_encoded, cv=skf, scoring='accuracy')

    print(f"Random Forest : {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
    print(f"MLP           : {mlp_scores.mean():.4f} ± {mlp_scores.std():.4f}")

    # ── Train final RF on full training set ──────────────────
    print(f"\n── Training final Random Forest ─────────────────────")
    rf.fit(X_train, y_train)

    # ── Test set evaluation ───────────────────────────────────
    y_pred = rf.predict(X_test)
    print("\n── Classification report (test set) ─────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names = le.classes_
    ))

    # ── Per-class accuracy — the honest metric ─────────────
    print("── Per-class accuracy ────────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    for i, cls in enumerate(le.classes_):
        row_total   = cm[i].sum()
        correct     = cm[i][i]
        per_class   = correct / row_total if row_total > 0 else 0
        bar         = "█" * int(per_class * 20)
        print(f"  {cls:10s}: {per_class:.3f}  {bar}")

    # ── Feature importance by finger ──────────────────────
    print("\n── Landmark importance (finger tips) ─────────────────")
    imp          = rf.feature_importances_
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    tip_landmarks= [4, 8, 12, 16, 20]   # landmark index for each tip

    for name, tip_lm in zip(finger_names, tip_landmarks):
        tip_feat = tip_lm * 3            # feature index (x of tip)
        tip_imp  = imp[tip_feat:tip_feat+3].sum()
        bar      = "█" * int(tip_imp * 200)
        print(f"  {name:8s} tip: {tip_imp:.4f}  {bar}")

    # ── Confusion matrix plot ─────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = 'd',
        cmap        = 'Blues',
        xticklabels = le.classes_,
        yticklabels = le.classes_
    )
    plt.title("Confusion matrix — gesture classifier")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(CM_IMAGE, dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved → {CM_IMAGE}")

    return rf, le


def save(model, encoder):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE,   'wb') as f: pickle.dump(model,   f)
    with open(ENCODER_FILE, 'wb') as f: pickle.dump(encoder, f)
    print(f"Model   saved → {MODEL_FILE}")
    print(f"Encoder saved → {ENCODER_FILE}")


def main():
    print("═" * 50)
    print("  Gesture Car — Model Training")
    print("═" * 50)

    X, y          = load_and_validate(DATA_CSV)
    model, encoder = train_and_evaluate(X, y)
    save(model, encoder)

    print("\nNext step: python src/controller.py")


if __name__ == "__main__":
    main()

    