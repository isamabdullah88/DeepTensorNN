
import os
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from model import DeepTensorNN
from data import getminiQM9

def get_torch_device() -> torch.device:
    """
    Detects the current platform and returns the best available
    PyTorch device based on hardware support.

    Returns:
        torch.device: The chosen device ('cuda', 'mps', or 'cpu').
    """
    system = platform.system()

    # --- macOS ---
    if system == "Darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("MPS not available — falling back to CPU.")
            return torch.device("cpu")

    # --- Windows / Linux ---
    elif system in ("Windows", "Linux"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available — using CPU.")
            return torch.device("cpu")

    # --- Fallback for unknown systems ---
    else:
        print(f"Unknown platform '{system}' — defaulting to CPU.")
        return torch.device("cpu")


def prepbatch(batch, numatoms, device):
    bsize = batch.num_graphs
    zblocks = []

    for i in range(bsize):
        zt = batch.z[batch.batch == i]
        n = zt.size(0)
        ztpadded = F.pad(zt, (0, numatoms-n), value=0.0)
        zblocks.append(ztpadded)
        
    zbatch = torch.stack(zblocks, dim=0)
    posbatch = torch.zeros(bsize, numatoms, 3).to(device)
    for i in range(bsize):
        idx = batch.batch == i
        posbatch[i, :idx.sum(), :] = batch.pos[idx, :]

    dist = torch.cdist(posbatch, posbatch)

    return zbatch, dist, batch.y[:, 7]

def train():
    import time
    from data import getminiQM9
    trainloader, valloader, testloader = getminiQM9()

    dfeatdim = 25
    numatoms = 10

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    writer = SummaryWriter()

    model = DeepTensorNN(numatoms, dfeatdim, 1)

    device = get_torch_device()
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    epochs = 100
    batch_size = 10
    for epoch in range(epochs+1):
        model.train()
        stime = time.time()
        for k, batch in enumerate(trainloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            zbatch, dist, target = prepbatch(batch, numatoms, device)
            pred = model(zbatch, dist)
            loss = criterion(pred, target)

            loss.backward()
            optimizer.step()

            writer.add_scalar('Batch-Loss/train', loss.item(), epoch+k)     

        # --- save model every 10 epochs ---
        if epoch % 10 == 0:
            checkpoint_path = f"checkpoints/model_E{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item()
            }, checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")

        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Time Taken: {(time.time()-stime): .01f}")

    writer.close()


if __name__ == "__main__":
    train()