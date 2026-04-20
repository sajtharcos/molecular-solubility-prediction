import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_excel("Chem/delaney-fixed.xlsx")

# Define the list of column names
cols = df.columns.tolist()

features_list = []
target_list = []

# Create the fingerprint generator once before the loop
from rdkit.Chem import rdFingerprintGenerator
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Use the generator inside the loop
for index, row in df.iterrows():
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol:
        # Directly generate bit vector
        fp = gen.GetFingerprint(mol)
        fp_list = list(fp)

        features_list.append(fp_list)
        target_list.append(row["measured log solubility in mols per litre"])

# Convert to PyTorch tensors
X = torch.tensor(features_list, dtype=torch.float32)
y = torch.tensor(target_list, dtype=torch.float32).view(-1, 1)

print("Data preparation completed.")

# Randomly split into 80% training and 20% testing data
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
        # Input -> 512 neurons -> 256 neurons -> 1 output
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)

model = SolubilityNet(input_dim=X.shape[1])

# Model, loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error (lower is better)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
train_losses = []

print("Training started...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Prediction and loss calculation
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backpropagation
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Training finished!")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)

    y_true = y_test.numpy()
    y_pred = predictions.numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f'\nTest loss: {test_loss.item():.4f}')
    print(f'MAE  = {mae:.4f}')
    print(f'RMSE = {rmse:.4f}')
    print(f'R²   = {r2:.4f}')

# Visualization: how well did we predict?
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Ideal line
plt.xlabel('True solubility')
plt.ylabel('Predicted solubility')
plt.title('True vs Predicted values')
plt.show()
