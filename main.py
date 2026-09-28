"""Diabetes prediction: PyTorch ANN vs scikit-learn Random Forest.

Run with:  python main.py
"""

# Importing libraries

import copy  #to make a snapshot (deep copy) of a model/config that won't change later
import os  #to work with files and folders (paths, checkpoints, datasets
import joblib # to save/load Python objects like encoders or scalers (joblib.dump / joblib.load)


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset


#data settings

CSV_PATH = "diabetes_prediction_dataset.csv"
TARGET = "diabetes"
SAVE_DIR = "models"


#ANN settings

BATCH_SIZE = 256
LEARNING_RATE = 1e-3 #0.001
MAX_EPOCHS = 10
PATIENCE = 10
DROPOUT_P = 0.3


SEED = 42 # makes splits and random weights init repeatable

os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# Loading CSV and inpecting data

df = pd.read_csv(CSV_PATH)
print(f"Shape:\n{df.shape}\n")
print("-" *60)
print(f"Dtypes:\n{df.dtypes}\n")
print("-" *60)
print(f"Sum of missing:\n{df.isna().sum()}")
print("-" *60)
print(f"Duplicate rows:\n{df.duplicated().sum()}")
print("-" *60)
print(f"Counting of target distributions:\n{df[TARGET].value_counts()}")
print("-" *60)
print(f"Fractions of target distribution:\n{df[TARGET].value_counts(normalize=True)}")
print("-" *60)
print(f"Smoking_history values:\n{df['smoking_history'].value_counts()}")
print("-" *60)


# Cleaning Data

rows_before = len(df)
print(rows_before)
df = df.drop_duplicates().reset_index(drop=True)
rows_after = len(df)


## Diagnostic only: counts rows with identical features but conflicting labels (no data modification).

feature_cols = [c for c in df.columns if c != TARGET]

same_features = df.duplicated(subset=feature_cols, keep=False)

n_conflict_rows = int(same_features.sum())

print(f"Rows whose features match another row but whose label may differ: "
      f"{n_conflict_rows} ({n_conflict_rows / len(df) * 100:.2f}% of data) -> kept")


#Spliting

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print(f"Train: {len(X_train)} rows | diabetes rate {y_train.mean():.4f}")
print(f"Val:   {len(X_val)} rows | diabetes rate {y_val.mean():.4f}")
print(f"Test:  {len(X_test)} rows | diabetes rate {y_test.mean():.4f}")


#Preprocessing

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

print(f"Numerical cols: {numeric_cols}")
print(f"Categorical cols: {categorical_cols}")


numeric_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),

])

categorical_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])


X_train_p = preprocessor.fit_transform(X_train)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)


# OneHotEncoder may return a sparse matrix & PyTorch needs a dense array

if hasattr(X_train_p, "toarray"):
    X_train_p = X_train_p.toarray()
    X_val_p = X_val_p.toarray()
    X_test_p = X_test_p.toarray()


#Rebuilding readable feature names

cat_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
    categorical_cols
).tolist()
feature_names = numeric_cols + cat_names


# float32 is the default dtype for PyTorch

X_train_p = X_train_p.astype(np.float32)
X_val_p = X_val_p.astype(np.float32)
X_test_p = X_test_p.astype(np.float32)
y_train_np = y_train.to_numpy().astype(np.float32)
y_val_np = y_val.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy().astype(np.float32)


print("\nProcessed shapes (rows, features):")
print("  train:", X_train_p.shape)
print("  val  :", X_val_p.shape)
print("  test :", X_test_p.shape)
print("Feature names after preprocessing:")
for i, name in enumerate(feature_names):
    print(f"  {i:2d}  {name}")


# Sanity checks: catch silent bugs early.

assert not np.isnan(X_train_p).any(), "NaN found in processed train data"
assert X_train_p.shape[1] == X_val_p.shape[1] == X_test_p.shape[1]
print("\nScaled numeric columns in TRAIN should have mean~0, std~1:")
n_num = len(numeric_cols)
print("  means:", X_train_p[:, :n_num].mean(axis=0).round(3))
print("  stds :", X_train_p[:, :n_num].std(axis=0).round(3))


joblib.dump(preprocessor, f"{SAVE_DIR}/preprocessor.joblib")


# Converting Data for Pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X_train_p.shape[1]
print(input_dim)


X_train_t = torch.from_numpy(X_train_p)
y_train_t = torch.from_numpy(y_train_np).unsqueeze(1)
X_val_t = torch.from_numpy(X_val_p)
y_val_t = torch.from_numpy(y_val_np).unsqueeze(1)
X_test_t = torch.from_numpy(X_test_p)
y_test_t = torch.from_numpy(y_test_np).unsqueeze(1)

print("X_train tensor shape:", tuple(X_train_t.shape), "| dtype:", X_train_t.dtype)
print("y_train tensor shape:", tuple(y_train_t.shape))


#Creating DataLoaders

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False
)


model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Dropout(DROPOUT_P),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(DROPOUT_P),
    nn.Linear(32, 1),
).to(device) 

n_params = sum(p.numel() for p in model.parameters())
print("Total learnable parameters:", n_params)


n_pos = float(y_train_np.sum())
n_neg = float(len(y_train_np) - n_pos)
pos_weight = torch.tensor([n_neg / n_pos], device=device)
print(f"\npos_weight = {pos_weight.item():.3f}  (negatives / positives in train)")
 
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#Training Loop

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss/len(loader.dataset)



def validate(model, loader, criterion, device):
    
    model.eval()
 
    total_loss = 0.0
 

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
 
    return total_loss / len(loader.dataset)

best_val_loss = float("inf")
best_weights = None
epochs_without_improvement = 0

print(f"{'epoch':>5} | {'train_loss':>10} | {'val_loss':>9} | note")


for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)

    note = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
       
        best_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        note = "<- best so far"

    else:
        epochs_without_improvement += 1

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:9.4f} | {note}")


    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}: no improvement for {PATIENCE} epochs.")
        break


model.load_state_dict(best_weights)
print(f"Restored best weights (val_loss = {best_val_loss:.4f})")

 



 


def predict_probabilities(model, X_tensor, device):
   
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor.to(device))
        probs = torch.sigmoid(logits)
    
    return probs.cpu().numpy().ravel()


def find_best_threshold(y_true, probs):
   
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score
 
 
def compute_metrics(y_true, probs, preds):
        
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1": f1_score(y_true, preds),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "Balanced Accuracy": balanced_accuracy_score(y_true, preds),
    }
 


def print_report(title, threshold, metrics, y_true, preds):
 
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Threshold        : {threshold:.2f}")
    for name, value in metrics.items():
        print(f"{name:<17}: {value:.4f}")

    cm = confusion_matrix(y_true, preds)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print("                pred 0   pred 1")
    print(f"  actual 0     {cm[0, 0]:7d}  {cm[0, 1]:7d}")
    print(f"  actual 1     {cm[1, 0]:7d}  {cm[1, 1]:7d}")


ann_val_probs = predict_probabilities(model, X_val_t, device)
ann_threshold, ann_val_bal_acc = find_best_threshold(y_val, ann_val_probs)
print(f"\nANN best threshold on validation set: {ann_threshold:.2f} "
      f"(balanced accuracy {ann_val_bal_acc:.4f})")


ann_test_probs = predict_probabilities(model, X_test_t, device)
ann_test_preds = (ann_test_probs >= ann_threshold).astype(int)
ann_metrics = compute_metrics(y_test, ann_test_probs, ann_test_preds)
print_report("ANN TEST RESULTS", ann_threshold, ann_metrics, y_test, ann_test_preds)


forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
)
 

forest.fit(X_train_p, y_train_np)
 
print("\n" + "=" * 60)
print("PART 11: RANDOM FOREST")
print("=" * 60)
print("Random Forest trained.")
 
rf_val_probs = forest.predict_proba(X_val_p)[:, 1]
rf_threshold, rf_val_bal_acc = find_best_threshold(y_val, rf_val_probs)
print(f"RF best threshold on validation set: {rf_threshold:.2f} "
      f"(balanced accuracy {rf_val_bal_acc:.4f})")
 
rf_test_probs = forest.predict_proba(X_test_p)[:, 1]
rf_test_preds = (rf_test_probs >= rf_threshold).astype(int)
rf_metrics = compute_metrics(y_test, rf_test_probs, rf_test_preds)
print_report("RANDOM FOREST TEST RESULTS", rf_threshold, rf_metrics, y_test, rf_test_preds)
 

importances = forest.feature_importances_
order = np.argsort(importances)[::-1]  
print("\nFeature importance (higher = used more by the forest):")
for i in order:
    print(f"  {feature_names[i]:28s} {importances[i]:.4f}")


    


print("\n" + "=" * 58)
print("MODEL COMPARISON ON THE TEST SET")
print("=" * 58)
print(f"{'Metric':<20} {'ANN':>10} {'RandomForest':>14} {'Better':>10}")
print("-" * 58)
for name in ann_metrics:
    a = ann_metrics[name]
    r = rf_metrics[name]
    if abs(a - r) < 0.0005:
        better = "tie"
    else:
        better = "ANN" if a > r else "RF"
    print(f"{name:<20} {a:10.4f} {r:14.4f} {better:>10}")
 
agree = (ann_test_preds == rf_test_preds).mean()
print(f"\nThe two models give the same prediction for {agree * 100:.2f}% of test patients.")
 

print(f"Test set contains {int(y_test.sum())} diabetic patients out of {len(y_test)}.")
print("Small metric gaps (roughly < 0.01) are usually noise, not a real winner.")

torch.save(
    {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "dropout_p": DROPOUT_P,
        "threshold": ann_threshold,
    },
    f"{SAVE_DIR}/ann_model.pt",
)
joblib.dump({"model": forest, "threshold": rf_threshold}, f"{SAVE_DIR}/random_forest.joblib")
print(f"\nSaved to '{SAVE_DIR}/': preprocessor.joblib, ann_model.pt, random_forest.joblib")
 

#Predicting new patients

print("\n" + "=" * 60)
print("PREDICT NEW PATIENTS (models reloaded from disk)")
print("=" * 60)
 
new_patients = pd.DataFrame([
    {   # a young, healthy-looking patient
        "gender": "Female", "age": 25.0, "hypertension": 0, "heart_disease": 0,
        "smoking_history": "never", "bmi": 22.5, "HbA1c_level": 4.8,
        "blood_glucose_level": 90,
    },
    {   # an older patient with high HbA1c and glucose
        "gender": "Male", "age": 62.0, "hypertension": 1, "heart_disease": 0,
        "smoking_history": "former", "bmi": 33.0, "HbA1c_level": 8.2,
        "blood_glucose_level": 220,
    },
    {   # borderline case
        "gender": "Female", "age": 50.0, "hypertension": 0, "heart_disease": 0,
        "smoking_history": "No Info", "bmi": 28.0, "HbA1c_level": 6.5,
        "blood_glucose_level": 140,
    },
])
print("New patients (raw):")
print(new_patients.to_string(index=False))
 

loaded_preprocessor = joblib.load(f"{SAVE_DIR}/preprocessor.joblib")
X_new = loaded_preprocessor.transform(new_patients)
if hasattr(X_new, "toarray"):
    X_new = X_new.toarray()
X_new = X_new.astype("float32")
 

checkpoint = torch.load(f"{SAVE_DIR}/ann_model.pt", map_location=device)
loaded_ann = nn.Sequential(
    nn.Linear(checkpoint["input_dim"], 64),
    nn.ReLU(),
    nn.Dropout(checkpoint["dropout_p"]),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(checkpoint["dropout_p"]),
    nn.Linear(32, 1),
).to(device)
loaded_ann.load_state_dict(checkpoint["state_dict"])
 
loaded_ann.eval()  
with torch.no_grad():
    new_logits = loaded_ann(torch.from_numpy(X_new).to(device))
    new_ann_probs = torch.sigmoid(new_logits).cpu().numpy().ravel()
new_ann_preds = (new_ann_probs >= checkpoint["threshold"]).astype(int)
 
rf_bundle = joblib.load(f"{SAVE_DIR}/random_forest.joblib")
new_rf_probs = rf_bundle["model"].predict_proba(X_new)[:, 1]
new_rf_preds = (new_rf_probs >= rf_bundle["threshold"]).astype(int)
 
print(f"\n{'patient':>7} | {'ANN prob':>8} {'ANN':>5} | {'RF prob':>8} {'RF':>5}")
print("-" * 50)
for i in range(len(new_patients)):
    ann_label = "YES" if new_ann_preds[i] == 1 else "no"
    rf_label = "YES" if new_rf_preds[i] == 1 else "no"
    print(f"{i + 1:>7} | {new_ann_probs[i]:8.3f} {ann_label:>5} | "
          f"{new_rf_probs[i]:8.3f} {rf_label:>5}")
 
print("\nNOTE: this is a learning project, not a medical device. The models")
print("were trained on one dataset and must not be used for real diagnoses.")
 
