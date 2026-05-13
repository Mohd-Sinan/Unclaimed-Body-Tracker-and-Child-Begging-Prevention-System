# Unclaimed-Body-Tracker-and-Child-Begging-Prevention-System( A College Final Year Project )

AI-powered missing persons, unidentified body, and child witness identification system with blockchain-style forensic audit verification.

## Features

* Face recognition for:

  * Missing persons
  * Missing children
  * Unidentified deceased individuals
  * Child witness reports
* Blockchain-style forensic audit chain
* Tamper detection using SHA256 manifests
* Match confirmation system
* Evidence verification
* Child witness case management
* Dataset integrity validation
* Match evidence storage
* Status tracking system
* Bootstrap recovery for existing datasets
* Similarity review dashboard
* Audit verification engine

---

# System Overview

CorpseID is designed as a forensic-grade identification platform.

The system maintains:

* Immutable audit records
* File integrity verification
* Evidence hashing
* Match traceability
* Case status history

Every critical operation generates a cryptographic audit block.

---

# Core Components

## Missing Persons Dataset

Stores:

* Metadata
* Face images
* Face embeddings

Verified using SHA256 manifest hashing.

---

## Missing Children Dataset

Dedicated child identification pipeline with:

* Child metadata
* Face embeddings
* Child-specific verification

---

## Child Witness Cases

Witnesses can upload:

* Child images
* Last seen information
* Event descriptions
* Contact information

Each case contains:

* `metadata.json`
* `status.json`
* Evidence images

Statuses are blockchain-audited.

---

## Confirmed Match Evidence

The system stores forensic evidence for:

* Confirmed dead matches
* Confirmed child matches

Evidence records are hashed and committed to the audit chain.

---

# Audit Chain

The system uses a blockchain-style append-only audit structure.

Each block contains:

```json
{
    "index": 1,
    "timestamp": "2026-01-01T00:00:00",
    "action": "CONFIRMED_CHILD_MATCH",
    "data_hash": "sha256_hash",
    "prev_hash": "previous_hash",
    "hash": "current_block_hash"
}
```

## Protected Operations

The audit chain verifies:

* Missing person records
* Missing child records
* Child witness cases
* Case status updates
* Confirmed dead matches
* Confirmed child matches
* Uploaded evidence images

---

# Tamper Detection

CorpseID detects:

* Modified metadata
* Deleted evidence
* Altered images
* Missing status files
* Corrupted manifests
* Chain manipulation

Verification functions:

```python
verify_chain()
verify_missing_dataset()
verify_child_dataset()
verify_child_cases()
verify_confirmed_dead_matches()
verify_child_matches()
```

---

# Tech Stack

## Backend

* Python
* Flask
* NumPy
* Pillow

## AI / Recognition

* Face embeddings
* Similarity comparison pipeline
* Dataset vector matching

## Storage

* JSON-based forensic manifests
* Local filesystem evidence storage
* SHA256 hashing

---

# Project Structure

```text
CorpseID/
│
├── app.py
├── train_adapter.py
├── audit_chain.py
├── storage/
│   ├── audit_chain.json
│   ├── confirmed_dead_matches/
│   └── confirmed_child_matches/
│
├── child_cases/
├── child_dataset/
├── missing_dataset/
├── missing_search_cases/
│
├── static/
└── templates
```

---

# Installation
I didn't account for the need for a requirements.txt since the entire project was completed in a total of 10 days during the last semester, where 90% of the work was done on the last 3 days before the final presentation. This took WAY too much effort on my part even though it was a group project, So i'm not going to bother with maintaining this project anymore. IF you have a Software Engineer job for me with a decent salary and perks, DO hire me.

## Run Application

```bash
python app.py
```

---

# Audit Verification

Run integrity verification:

```python
verify_chain()
```

Verify all forensic datasets:

```python
verify_missing_dataset()
verify_child_dataset()
verify_child_cases()
verify_confirmed_dead_matches()
verify_child_matches()
```

---

# Security Model

CorpseID uses:

* SHA256 evidence hashing
* Append-only audit records
* Manifest-based verification
* Linked block hashes
* Evidence integrity validation

The system is designed to make silent forensic tampering detectable.

---

## Adapter Layer

CorpseID includes a custom machine-learning adapter layer designed to improve identity classification on top of raw facial embeddings.

Instead of relying purely on cosine similarity, the system trains a secondary statistical classifier that learns identity separation patterns from the embedding space.

---

## Adapter Architecture

### Embedding Source

The adapter consumes embeddings generated from:

* DeepFace
* FaceNet512
* RetinaFace detection pipeline

Each face embedding is stored as a `.npy` vector.

---

## Adapter Training Pipeline

### 1. Dataset Loading

The adapter scans the dataset directory recursively and loads all saved embedding vectors.

```python
for person_id in os.listdir(DATASET):
```

Every embedding is linked to its corresponding identity label.

---

### 2. L2 Normalization

All embeddings are normalized before processing.

```python
emb = emb / np.linalg.norm(emb)
```

This stabilizes vector magnitude differences and improves classification consistency.

---

### 3. Feature Engineering

The adapter expands every embedding into a larger forensic feature space.

Generated features:

* original embedding
* squared embedding
* absolute embedding

```python
feature_vector = np.concatenate([
    emb,
    emb ** 2,
    np.abs(emb)
])
```

For FaceNet512:

* 512 original features
* 512 squared features
* 512 absolute features

Final feature dimensionality:

```text
1536-dimensional forensic feature vector
```

This improves nonlinear identity separation beyond traditional cosine similarity.

---

### 4. Feature Standardization

The adapter standardizes the expanded feature space using:

```python
StandardScaler()
```

This prevents feature imbalance before dimensionality reduction.

---

### 5. PCA Dimensionality Reduction

Principal Component Analysis (PCA) is applied to:

* reduce redundancy
* compress feature space
* remove noise
* improve generalization

Configuration:

```python
PCA(
    n_components=0.95,
    whiten=True,
    random_state=42
)
```

The system automatically preserves 95% of feature variance.

---

### 6. Validation Detection

The adapter automatically checks whether validation is statistically possible.

Validation is enabled only when:

* every identity class contains at least two samples
* multiple identity classes exist

This prevents invalid benchmark metrics on tiny forensic datasets.

---

### 7. Adapter Classifier

The final identity classifier uses multinomial logistic regression.

```python
LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=0.7,
    solver="lbfgs",
    multi_class="multinomial",
    n_jobs=-1
)
```

### Why Logistic Regression?

This model provides:

* fast inference
* probabilistic identity prediction
* stable multiclass classification
* low deployment complexity
* strong performance on normalized embeddings

---

### 8. Performance Metrics

When validation is available, the adapter computes:

* accuracy
* precision
* recall
* F1 score
* confusion matrix

This allows forensic benchmarking of identity separation quality.

---

### 9. Pipeline Serialization

The complete adapter pipeline is serialized using Joblib.

Saved components include:

* trained classifier
* feature scaler
* PCA reducer

```python
joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "pca": pca
    },
    "arcface_adapter.pkl"
)
```

---

## Purpose of the Adapter Layer

The adapter layer acts as a forensic identity refinement engine between raw embeddings and final match decisions.

It improves:

* identity separation
* large-dataset scalability
* probabilistic matching
* false-positive resistance
* forensic consistency
* embedding-space discrimination

while remaining fully modular and replaceable.

---

# Important Notes

This project is intended for:

* Research
* Digital forensics experimentation
* Missing person investigation tooling
* Integrity verification research
* AI-assisted identification systems

It is not a replacement for official law enforcement systems.

---

# License

MIT License

---

# Author

Mohammed Sinan KH
README generated using Chatgpt
