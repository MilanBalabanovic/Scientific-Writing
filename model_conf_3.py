import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import time


# --- Load datasets ---
train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")

for df in [train_df, test_df]:
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

# --- Separate target and remove attack_cat ---
y_train = train_df['label'].astype(int)
y_test = test_df['label'].astype(int)

# Remove BOTH 'label' and 'attack_cat' before encoding
X_train = train_df.drop(columns=['label', 'attack_cat'], errors='ignore')
X_test = test_df.drop(columns=['label', 'attack_cat'], errors='ignore')

# --- Encode categorical features ---
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align train/test columns
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --- Scale numeric features ---
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Convert to Torch tensors ---
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

# --- Define the model ---
class IDSNet(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- Initialize ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train.shape[1]
model = IDSNet(input_dim=input_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# --- Train ---
epochs = 50
train_losses, val_losses, val_accs = [], [], []
train_start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, preds, true = 0, [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            true.extend(yb.cpu().numpy())
    val_loss /= len(test_loader.dataset)
    acc = accuracy_score(true, preds)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(acc)

    print(f"Epoch {epoch+1:02d}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {acc:.4f}")

train_duration = time.time() - train_start_time

# --- Metrics ---
cm = confusion_matrix(true, preds)
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print("\n--- Final Evaluation ---")
print(f"Accuracy: {accuracy_score(true, preds):.4f}")
print(f"Detection Rate (TPR): {tpr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"Training duration (seconds): {train_duration:.2f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(true, preds, digits=4))

# --- Plot ---
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.ylim(0.75, 0.95)
plt.subplot(1,2,2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.ylim(0.1, 0.35)
plt.tight_layout(); plt.show()
