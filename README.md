# Molecular Solubility Prediction

This project explores machine learning approaches for predicting molecular solubility (logS) from SMILES representations.

The goal is to compare different molecular representations and modeling strategies, including:
- descriptor-based neural networks
- Morgan fingerprint-based models
- graph neural networks (GNNs)

The project uses the Delaney (ESOL) dataset, which contains experimentally measured solubility values for small organic molecules.

## Key ideas
- Converting SMILES strings into numerical representations
- Comparing handcrafted descriptors vs. learned representations
- Evaluating model performance using MAE, RMSE, and R²
- Understanding how molecular structure influences solubility

## Technologies
- Python
- PyTorch
- RDKit
- scikit-learn
- PyTorch Geometric (for GNN models)

## Goal
The main objective is not only to build predictive models, but also to understand how different representations affect model performance in cheminformatics tasks.

## Results

| Model | MAE | RMSE | R² |
|------|-----|------|----|
| Descriptor-based NN (vol1) | 0.5405 | 0.7085 | 0.8795 |
| Morgan fingerprint NN (vol2) | 0.9072 | 1.1747 | 0.6591 |
| Hybrid (fingerprint + descriptors, vol3) | 0.6393 | 0.9151 | 0.8214 |
