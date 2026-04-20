import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load dataset
df = pd.read_excel("Chem/delaney-fixed.xlsx")

# 2. Define list of column names
cols = df.columns.tolist()

features_list = []
target_list = []

for index, row in df.iterrows():
    # Compute chemical features from SMILES if needed
    mol  = Chem.MolFromSmiles(row["smiles"])
    if mol:
        # Define features
        # If available in Excel, use them, otherwise compute with RDKit
        
        # 1. Molecular weight
        mw = row["Molecular Weight"] if "Molecular Weight" in cols else Descriptors.MolWt(mol)
        
        # 2. H-bond donors
        h_donors = row["Number of H-Bond Donors"] if "Number of H-Bond Donors" in cols else Descriptors.NumHDonors(mol)
        
        # 3. LogP (lipophilicity) - not included in Excel by default, compute it
        logp = Descriptors.MolLogP(mol)
        
        # 4. Number of rings
        rings = row["Number of Rings"] if "Number of Rings" in cols else Descriptors.CalcNumRings(mol)

        features_list.append([mw, h_donors, logp, rings])
        
        # Target value (assumed to always exist)
        target_col = "measured log solubility in mols per litre"
        target_list.append(row[target_col])

# Convert to PyTorch tensors
X = torch.tensor(features_list, dtype=torch.float32)
y = torch.tensor(target_list, dtype=torch.float32).view(-1, 1)

# Normalization (Z-score scaling): (x - mean) / std
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X = (X - X_mean) / X_std

print("Data normalization completed.")

# Randomly split into 80% training and 20% test data
indices = torch.randperm(X.size(0))
train_size = int(0.8 * X.size(0))

train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

class SolubilityNet(nn.Module):
    def __init__(self, input_dim):
        super(SolubilityNet, self).__init__()
        # 4 inputs -> 16 neurons -> 8 neurons -> 1 output
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

model = SolubilityNet(input_dim=X.shape[1])

# 1. Model, loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error (lower is better)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Optimizer ("teacher")

# 2. Training loop
epochs = 1000
train_losses = []

print("Training started...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Reset gradients
    
    # Prediction and loss calculation
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backpropagation (learning happens here)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training finished!")

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f'\nTest loss: {test_loss.item():.4f}')

# Visualization: how well did we predict?
plt.figure(figsize=(8,6))
plt.scatter(y_test.numpy(), predictions.numpy(), alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
plt.xlabel('True solubility')
plt.ylabel('Predicted solubility')
plt.title('True vs Predicted values')
plt.show()