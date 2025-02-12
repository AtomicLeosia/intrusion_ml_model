import os
import sys
import pyfiglet
import numpy as np
import pandas as pd
import train_test_libs as ttl
import download_dataset
import validate_integrity
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from def_model_mlp import ModelMLP, ModelDeepMLP
from def_model_cnn import ModelCNN, ModelDeepCNN
from def_model_ae import ModelAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    ascii_art = pyfiglet.figlet_format("RTX On", font="small")
    print(ascii_art)
else:
    print()

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

model_mlp_path = "model/model_mlp.pth"
model_cnn_path = "model/model_cnn.pth"

if not os.path.isdir("dataset") : download_dataset.retrieve()

if validate_integrity.process_dir("dataset", "checksum"):
    print("\nAll dataset integrity checks passed.")
else: # Alteration could produce unreliable results. So no.
    print("\nDataset integrity checks failed! Aborting...")
    sys.exit(-1)

df = ttl.load_set(cool_days)

###############################################################################
# 1. Data cleanup and preprocessing
###############################################################################

counts = df["Label"].value_counts()
rare_classes = counts[counts < 1000].index
df = df[~df["Label"].isin(rare_classes)]

df.drop_duplicates(inplace=True)
df.replace(np.inf, np.nan, inplace=True)
df.dropna(inplace=True)

le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
original_labels = list(label_mapping.keys())

X = df.drop(columns=["Label"])
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, # Keeps it consistent
    stratify=y)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

batch_size = 256 # Even though entire dataset fits in VRAM, this is optimal

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = X_train.shape[1]
num_classes = len(label_mapping)

labels = np.array(list(label_mapping.values()))

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=labels,
    y=df["Label"].values)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

###############################################################################
# 2. MLP
###############################################################################
#"""
model_mlp = ModelMLP(input_dim, num_classes).to(device)
criterion_mlp = nn.CrossEntropyLoss(weight=class_weights)

model_mlp_state = torch.load(model_mlp_path, map_location=device)
model_mlp.load_state_dict(model_mlp_state['model_state_dict'], strict=True) # Require keys match
last_epoch = model_mlp_state['train_state_epoch']

print("\n======================== Begin MLP Model Evaluation ========================")

test_loss, test_acc, test_precision, test_recall, test_preds, test_labels = ttl.test_one_epoch(
    model_mlp, test_loader, criterion_mlp)

print(f"Last epoch: {last_epoch:02d}, "
          f"Loss: {test_loss:.4f}, "
          f"Acc: {test_acc:.4f}, "
          f"Precision: {test_precision:.4f}, "
          f"Recall: {test_recall:.4f}")

test_labels_original = [original_labels[i] for i in test_labels.numpy()]
test_preds_original = [original_labels[i] for i in test_preds.numpy()]

cm_mlp = confusion_matrix(test_labels_original, test_preds_original)
cm_mlp_df = pd.DataFrame(cm_mlp, index=original_labels, columns=original_labels)
report_mlp = classification_report(test_labels_original, test_preds_original, zero_division=0)

print("\nConfusion Matrix:\n", cm_mlp_df)
print("\nClassification Report:\n", report_mlp)

del model_mlp ## reconsider
torch.cuda.empty_cache()  # Free up GPU memory

###############################################################################
# 2. CNN
###############################################################################
#"""
model_cnn = ModelCNN(input_dim, num_classes).to(device)
criterion_cnn = nn.CrossEntropyLoss(weight=class_weights)

model_cnn_state = torch.load(model_cnn_path, map_location=device)
model_cnn.load_state_dict(model_cnn_state['model_state_dict'], strict=True) # Require keys match
last_epoch = model_cnn_state['train_state_epoch']

print("\n======================== Begin CNN Model Evaluation ========================")

test_loss, test_acc, test_precision, test_recall, test_preds, test_labels = ttl.test_one_epoch(
    model_cnn, test_loader, criterion_cnn)

print(f"Last epoch: {last_epoch:02d}, "
          f"Loss: {test_loss:.4f}, "
          f"Acc: {test_acc:.4f}, "
          f"Precision: {test_precision:.4f}, "
          f"Recall: {test_recall:.4f}")

test_labels_original = [original_labels[i] for i in test_labels.numpy()]
test_preds_original = [original_labels[i] for i in test_preds.numpy()]

cm_cnn = confusion_matrix(test_labels_original, test_preds_original)
cm_cnn_df = pd.DataFrame(cm_cnn, index=original_labels, columns=original_labels)
report_cnn = classification_report(test_labels_original, test_preds_original, zero_division=0)

print("\nConfusion Matrix:\n", cm_cnn_df)
print("\nClassification Report:\n", report_cnn)
