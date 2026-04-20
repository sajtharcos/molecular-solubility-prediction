import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# 2. Load data
# =========================================================
df = pd.read_excel("Chem/delaney-fixed.xlsx")

# Target column
target_col = "measured log solubility in mols per litre"

# Numerical descriptor columns
descriptor_cols = [
    "ESOL predicted log solubility in mols per litre",
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area"
]

# Keep only required columns
needed_cols = ["smiles", target_col] + descriptor_cols
df = df[needed_cols].copy()

# Drop missing values
df = df.dropna(subset=needed_cols).reset_index(drop=True)

print(f"Number of samples used: {len(df)}")

# =========================================================
# 3. Morgan fingerprint generator
# =========================================================
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

fingerprints = []
descriptors = []
targets = []

for _, row in df.iterrows():
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    # Compute Morgan fingerprint
    fp = gen.GetFingerprint(mol)
    fp_array = np.array(list(fp), dtype=np.float32)

    # Extract descriptors from table
    desc_array = row[descriptor_cols].values.astype(np.float32)

    # Target value
    y_value = np.float32(row[target_col])

    fingerprints.append(fp_array)
    descriptors.append(desc_array)
    targets.append(y_value)

fingerprints = np.array(fingerprints, dtype=np.float32)
descriptors = np.array(descriptors, dtype=np.float32)
targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

print(f"Valid RDKit molecules: {len(targets)}")

# =========================================================
# 4. Normalize descriptors
#    (fingerprints are binary → no scaling needed)
# =========================================================
desc_mean = descriptors.mean(axis=0)
desc_std = descriptors.std(axis=0)

# Prevent division by zero
desc_std[desc_std == 0] = 1.0

descriptors_scaled = (descriptors - desc_mean) / desc_std

# Concatenate fingerprint + descriptors
X = np.concatenate([fingerprints, descriptors_scaled], axis=1)
y = targets

print(f"Input dimension: {X.shape[1]}")

# =========================================================
# 5. Train / Validation / Test split
# =========================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=SEED
)
# ~70% train, 15% val, 15% test

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# =========================================================
# 6. Model
# =========================================================
class SolubilityNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

model = SolubilityNet(input_dim=X_train.shape[1])

criterion = nn.SmoothL1Loss()  # More robust than MSE
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

# =========================================================
# 7. Training with early stopping
# =========================================================
epochs = 1000
patience = 50

best_val_loss = float("inf")
best_state = None
patience_counter = 0

train_losses = []
val_losses = []

print("Training started...")

for epoch in range(epochs):
    # ---- training ----
    model.train()
    optimizer.zero_grad()

    train_pred = model(X_train)
    train_loss = criterion(train_pred, y_train)

    train_loss.backward()
    optimizer.step()

    # ---- validation ----
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train loss: {train_loss.item():.4f} | "
            f"Val loss: {val_loss.item():.4f}"
        )

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}.")
        break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

print("Training finished.")

# =========================================================
# 8. Evaluation
# =========================================================
model.eval()
with torch.no_grad():
    test_pred = model(X_test).cpu().numpy()
    y_true = y_test.cpu().numpy()

mae = mean_absolute_error(y_true, test_pred)
rmse = np.sqrt(mean_squared_error(y_true, test_pred))
r2 = r2_score(y_true, test_pred)

print("\nTest results:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# =========================================================
# 9. Plots
# =========================================================

# Learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning curve")
plt.legend()
plt.grid(True)
plt.show()

# Real vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_true, test_pred, alpha=0.6)
min_val = min(y_true.min(), test_pred.min())
max_val = max(y_true.max(), test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--")
plt.xlabel("True solubility")
plt.ylabel("Predicted solubility")
plt.title("True vs Predicted")
plt.grid(True)
plt.show()