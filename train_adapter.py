"""import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

DATASET = "dataset"

X = []
y = []

for person_id in os.listdir(DATASET):

    folder = os.path.join(DATASET, person_id)

    for file in os.listdir(folder):
        if file.endswith(".npy"):

            emb_path = os.path.join(folder, file)
            emb = np.load(emb_path)

            # normalize
            emb = emb / np.linalg.norm(emb)

            X.append(emb)
            y.append(person_id)

X = np.array(X)

print("Training adapter layer...")

model = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)

model.fit(X, y)

joblib.dump(model, "arcface_adapter.pkl")

print("Adapter saved.")"""

import os
import numpy as np
import joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.decomposition import PCA

DATASET = "dataset"

# =========================================
# STORAGE
# =========================================
X = []
y = []

print("\n==============================")
print(" LOADING DATASET ")
print("==============================")

# =========================================
# LOAD ALL EMBEDDINGS
# =========================================
for person_id in os.listdir(DATASET):

    person_folder = os.path.join(DATASET, person_id)

    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):

        if file.endswith(".npy"):

            emb_path = os.path.join(person_folder, file)

            emb = np.load(emb_path)

            # =========================================
            # L2 NORMALIZATION
            # =========================================
            emb = emb / np.linalg.norm(emb)

            # =========================================
            # FEATURE ENGINEERING
            # =========================================
            #
            # original embedding
            # squared embedding
            # absolute embedding
            #
            # final dimensionality:
            # 512 * 3 = 1536
            #
            feature_vector = np.concatenate([
                emb,
                emb ** 2,
                np.abs(emb)
            ])

            X.append(feature_vector)
            y.append(person_id)

X = np.array(X)
y = np.array(y)

print(f"Total Samples : {len(X)}")
print(f"Total Classes : {len(set(y))}")
print(f"Feature Shape : {X.shape}")

# =========================================
# STANDARDIZATION
# =========================================
print("\n==============================")
print(" STANDARDIZING FEATURES ")
print("==============================")

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =========================================
# PCA DIMENSIONALITY REDUCTION
# =========================================
print("\n==============================")
print(" APPLYING PCA ")
print("==============================")

pca = PCA(
    n_components=0.95,
    whiten=True,
    random_state=42
)

X = pca.fit_transform(X)

print("Reduced Feature Shape:", X.shape)

# =========================================
# TRAIN / TEST SPLIT
# =========================================
print("\n==============================")
print(" TRAIN / TEST SPLIT ")
print("==============================")

class_counts = Counter(y)

# validation possible only if every class has >=2 samples
can_validate = all(count >= 2 for count in class_counts.values())

if can_validate and len(set(y)) > 1:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Validation Enabled")

else:

    print("Validation skipped (single-sample classes detected)")

    X_train = X
    y_train = y
    X_test = None
    y_test = None

print("Training Samples :", len(X_train))

if X_test is not None:
    print("Testing Samples  :", len(X_test))
else:
    print("Testing Samples  : Validation Skipped")

# =========================================
# MODEL
# =========================================
print("\n==============================")
print(" TRAINING ADAPTER ")
print("==============================")

model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=0.7,
    solver="lbfgs",
    multi_class="multinomial",
    n_jobs=-1
)

# =========================================
# TRAINING
# =========================================
model.fit(X_train, y_train)

print("Training Complete.")

# =========================================
# VALIDATION
# =========================================
print("\n==============================")
print(" VALIDATION ")
print("==============================")

train_preds = model.predict(X_train)

if X_test is not None:
    test_preds = model.predict(X_test)

# =========================================
# METRICS
# =========================================
train_acc = accuracy_score(y_train, train_preds)

print("\n===== PERFORMANCE METRICS =====")
print(f"Train Accuracy : {train_acc * 100:.2f}%")

if X_test is not None:

    test_acc = accuracy_score(y_test, test_preds)

    precision = precision_score(
        y_test,
        test_preds,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_preds,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_preds,
        average="weighted",
        zero_division=0
    )

    print(f"Test Accuracy  : {test_acc * 100:.2f}%")
    print(f"Precision      : {precision * 100:.2f}%")
    print(f"Recall         : {recall * 100:.2f}%")
    print(f"F1 Score       : {f1 * 100:.2f}%")

else:

    print("Validation metrics unavailable.")
# =========================================
# CONFUSION MATRIX
# =========================================
if X_test is not None:

    print("\n==============================")
    print(" CONFUSION MATRIX ")
    print("==============================")

    cm = confusion_matrix(y_test, test_preds)

    print(cm)
# =========================================
# SAVE FULL PIPELINE
# =========================================
print("\n==============================")
print(" SAVING PIPELINE ")
print("==============================")

joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "pca": pca
    },
    "arcface_adapter.pkl"
)

print("Pipeline Saved Successfully.")
print("\nTraining Finished.\n")
