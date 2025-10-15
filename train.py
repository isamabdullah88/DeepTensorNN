from model import DeepTensorNN
from data import getQM9data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def prepbatch(batch, numatoms, device):
    bsize = batch.num_graphs
    zblocks = []

    for i in range(bsize):
        zt = batch.z[batch.batch == i]
        n = zt.size(0)
        ztpadded = F.pad(zt, (0, numatoms-n), value=0.0)
        zblocks.append(ztpadded)
        
    # zbatch = pad_sequence(zblocks, batch_first=True)
    zbatch = torch.stack(zblocks, dim=0)
    # print('zbatch: ', zbatch.size())
    # print('pos: ', batch.pos.size())
    posbatch = torch.zeros(bsize, numatoms, 3).to(device)
    for i in range(bsize):
        idx = batch.batch == i
        posbatch[i, :idx.sum(), :] = batch.pos[idx, :]

    # print('posbatch: ', posbatch.size())
    dist = torch.cdist(posbatch, posbatch)

    return zbatch, dist, batch.y[:, 7]

def train():
    import time
    from data import getQM9data
    trainloader, valloader, testloader = getQM9data()

    dfeatdim = 25
    numatoms = 10

    writer = SummaryWriter()

    model = DeepTensorNN(numatoms, dfeatdim, 1)

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
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
            
            # print('bsize: ', bsize)
            # n = batch.pos.size(0)
            
            
            # zblocks = []
            # for i in range(bsize):
            #     zt = batch.z[batch.batch == i]
            #     n = zt.size(0)
            #     ztpadded = F.pad(zt, (0, numatoms-n), value=0.0)
            #     zblocks.append(ztpadded)
                
            # # zbatch = pad_sequence(zblocks, batch_first=True)
            # zbatch = torch.stack(zblocks, dim=0)
            # # print('zbatch: ', zbatch.size())
            # # print('pos: ', batch.pos.size())
            # posbatch = torch.zeros(bsize, numatoms, 3).to(device)
            # for i in range(bsize):
            #     idx = batch.batch == i
            #     posbatch[i, :idx.sum(), :] = batch.pos[idx, :]

            # # print('posbatch: ', posbatch.size())
            # dist = torch.cdist(posbatch, posbatch)
            # print('dist: ', dist.size())
            # print('y: ', batch.y[:, 7].size())
            zbatch, dist, target = prepbatch(batch, numatoms, device)
            pred = model(zbatch, dist)
            loss = criterion(pred, target)

            # loss = loss / batch_size

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