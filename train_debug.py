import os
import sys
import pyfiglet
import debug_libs as dl
import train_test_libs as ttl
import download_dataset
import validate_integrity
from def_model_mlp import ModelMLP, ModelDeepMLP
from def_model_cnn import ModelCNN, ModelDeepCNN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    ascii_art = pyfiglet.figlet_format("RTX On", font="small")
    print(ascii_art)

full_dataset = [
    "dataset/Monday-WorkingHours.pcap_ISCX.csv",
    "dataset/Tuesday-WorkingHours.pcap_ISCX.csv",
    "dataset/Wednesday-workingHours.pcap_ISCX.csv",
    "dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

lame_days = [
    "dataset/Monday-WorkingHours.pcap_ISCX.csv",
    "dataset/Tuesday-WorkingHours.pcap_ISCX.csv",
    "dataset/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Morning.pcap_ISCX.csv",
]

cool_days = [
    "dataset/Wednesday-workingHours.pcap_ISCX.csv",
    "dataset/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "dataset/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

if not os.path.isdir("dataset") : download_dataset.retrieve()

if validate_integrity.process_dir("dataset", "checksum"):
    print("\nAll dataset integrity checks passed.")
else:
    print("\nDataset integrity checks failed! Aborting...")
    sys.exit(-1)

os.makedirs("metrics", exist_ok=True)
os.makedirs("model", exist_ok=True)

df = ttl.load_set(cool_days)

print(f"\nTotal records in dataset: {df.shape[0]}")
print(f"Total number of columns: {df.shape[1]}")

###############################################################################
# 1. Data cleanup and preprocessing
###############################################################################

# Drop rare classes (less than 1000 examples)
counts = df["Label"].value_counts()
rare_classes = counts[counts < 1000].index
df = df[~df["Label"].isin(rare_classes)]

dl.label_distrib(df)

print("\nDropping", df.duplicated().sum(), "duplicate rows...")
df.drop_duplicates(inplace=True)

dl.label_distrib(df)
dl.count_invalid(df)

# Replace inf with NaN, then drop
df.replace(np.inf, np.nan, inplace=True)

invalid_rows_mask = df.isna().any(axis=1)
invalid_rows = df[invalid_rows_mask]
label_count = invalid_rows["Label"].value_counts()
print("\nInvalid row count by corresponding label:\n\n", label_count)

print("\nDropping", invalid_rows_mask.sum(), "invalid value rows...")
df.dropna(inplace=True)

print("\nMissing value row count:", df.isnull().sum().sum())

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", cat_cols)

# Label becomes numerical, but we keep the mapping
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))   ### Le classes xddd
original_labels = list(label_mapping.keys())

X = df.drop(columns=["Label"])
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Tried SMOTE, didn't work. The opposite, in fact.
"""
print("\nPerforming SMOTE oversampling...")
warnings.simplefilter(action='ignore', category=FutureWarning)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled.values, dtype=torch.long)
"""
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

batch_size = 256 # Even though entire dataset fits in VRAM, this is optimal

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = X_train.shape[1]
num_classes = len(label_mapping)

# Weighted loss setup. Used in criterion functions
labels = np.array(list(label_mapping.values()))
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=labels,
    y=df["Label"].values
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

###############################################################################
# 2. MLP
###############################################################################
#"""
mlp_csv_file = "metrics/mlp_metrics.csv"
dl.init_log(mlp_csv_file)

model_mlp = ModelMLP(input_dim, num_classes).to(device)
criterion_mlp = nn.CrossEntropyLoss(weight=class_weights)
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.001)
scheduler_mlp = torch.optim.lr_scheduler.ReduceLROnPlateau( # Adaptywny
    optimizer_mlp, mode='min', factor=0.5, patience=3, verbose=True)

epochs_mlp = 30

print("\n========================== Begin MLP Model Training =========================")

for epoch in range(1, epochs_mlp + 1):

    train_loss, train_acc, train_precision, train_recall = ttl.train_one_epoch(
        model_mlp, train_loader, criterion_mlp, optimizer_mlp)

    test_loss, test_acc, test_precision, test_recall, test_preds, test_labels = ttl.test_one_epoch(
        model_mlp, test_loader, criterion_mlp)

    print(f"Epoch [{epoch:02d}/{epochs_mlp:02d}], "
          f"Loss: {train_loss:.4f}, "
          f"Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Acc: {test_acc:.4f}")

    dl.log_metrics(mlp_csv_file, epoch,
        train_loss, train_acc, train_precision, train_recall,
        test_loss, test_acc, test_precision, test_recall)
    
    scheduler_mlp.step(test_loss)

torch.save({
    'train_state_loss': train_loss,
    'train_state_epoch': epochs_mlp,
    'model_state_dict': model_mlp.state_dict(),
    'optimizer_state_dict': optimizer_mlp.state_dict(),
}, "model/model_mlp.pth")

# Covert labels back to strings
test_labels_original = [original_labels[i] for i in test_labels.numpy()]
test_preds_original = [original_labels[i] for i in test_preds.numpy()]

cm_mlp = confusion_matrix(test_labels_original, test_preds_original)
cm_mlp_df = pd.DataFrame(cm_mlp, index=original_labels, columns=original_labels)
report_mlp = classification_report(test_labels_original, test_preds_original, zero_division=0)

print("\nConfusion Matrix:\n", cm_mlp_df)
print("\nClassification Report:\n", report_mlp)
#"""
###############################################################################
# 3. CNN
###############################################################################
#"""
cnn_csv_file = "metrics/final_cnn_metrics.csv"
dl.init_log(cnn_csv_file)

model_cnn = ModelCNN(input_dim, num_classes).to(device)
criterion_cnn = nn.CrossEntropyLoss(weight=class_weights)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)
scheduler_cnn = torch.optim.lr_scheduler.StepLR( # O połowę co 10 epok
    optimizer_cnn, step_size=10, gamma=0.5)

epochs_cnn = 45

print("\n========================== Begin CNN Model Training =========================")

for epoch in range(1, epochs_cnn + 1):

    train_loss, train_acc, train_precision, train_recall = ttl.train_one_epoch(
    model_cnn, train_loader, criterion_cnn, optimizer_cnn)

    test_loss, test_acc, test_precision, test_recall, test_preds, test_labels = ttl.test_one_epoch(
    model_cnn, test_loader, criterion_cnn)

    print(f"Epoch [{epoch:02d}/{epochs_cnn:02d}], "
          f"Loss: {train_loss:.4f}, "
          f"Acc: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Acc: {test_acc:.4f}")

    dl.log_metrics(cnn_csv_file, epoch,
                train_loss, train_acc, train_precision, train_recall,
                test_loss, test_acc, test_precision, test_recall)
    
    scheduler_cnn.step()

torch.save({
    'train_state_loss': train_loss,
    'train_state_epoch': epochs_cnn,
    'model_state_dict': model_cnn.state_dict(),
    'optimizer_state_dict': optimizer_cnn.state_dict(),
}, "model/model_cnn.pth")

test_labels_original = [original_labels[i] for i in test_labels.numpy()]
test_preds_original = [original_labels[i] for i in test_preds.numpy()]

cm_cnn = confusion_matrix(test_labels_original, test_preds_original)
cm_cnn_df = pd.DataFrame(cm_cnn, index=original_labels, columns=original_labels)
report_cnn = classification_report(test_labels_original, test_preds_original, zero_division=0)

print("\nConfusion Matrix:\n", cm_cnn_df)
print("\nClassification Report:\n", report_cnn)
#"""