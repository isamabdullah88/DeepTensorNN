import torch
from model import DeepTensorNN

from data import getQM9data
from train import prepbatch

# Example: path to saved model
checkpoint_path = "checkpoints/model_E60.pt"

device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")

dfeatdim = 25
numatoms = 10
# Load model architecture
model = DeepTensorNN(numatoms, dfeatdim, 1)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

model = model.to(device)


_, _, testloader = getQM9data()

predictions, targets = [], []

with torch.no_grad():
    for batch in testloader:
        batch = batch.to(device)
        # for mol_id in batch.batch.unique():
        #     mol_mask = (batch.batch == mol_id)
        #     z_mol = batch.z[mol_mask]
        #     # molecules_z.append(z_mol)
        #     pos = batch.pos[batch.batch == mol_id]
        #     dist = torch.cdist(pos, pos)
        #     e = batch.y[mol_id, 7]
            
        #     pred = model(z_mol, dist)

        #     predictions.append(pred.cpu())
        #     targets.append(e.cpu())
        zbatch, dist, target = prepbatch(batch, numatoms, device)

        pred = model(zbatch, dist)

        predictions.append(pred)
        targets.append(target)

predictions = torch.concatenate(predictions).cpu()
targets = torch.concatenate(targets).cpu()

mae = torch.mean(torch.abs(predictions - targets))
rmse = torch.sqrt(torch.mean((predictions - targets)**2))
r2 = 1 - torch.sum((predictions - targets)**2) / torch.sum((targets - targets.mean())**2)

print(f"MAE: {mae.item():.4f} eV")
print(f"RMSE: {rmse.item():.4f} eV")
print(f"R²: {r2.item():.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(targets, predictions, s=10, alpha=0.7)
plt.plot([targets.min(), targets.max()],
         [targets.min(), targets.max()],
         'r--', label='Perfect prediction')

plt.xlabel("True Energy (eV)")
plt.ylabel("Predicted Energy (eV)")
plt.title("Predicted vs True Total Energy")
plt.legend()
plt.show()