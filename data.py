
import os
import torch
from torch_geometric.data import Dataset

class MiniQM9(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self.processed_file = os.path.join(root, "processed", "miniQM9.pt")
        self.data_list = torch.load(self.processed_file, weights_only=False)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
    
def normalize(data):
    energies = torch.tensor([d.y[:, 7].item() for d in data])

    mean = energies.mean()
    std = energies.std()

    for i,d in enumerate(data):
        data[i].y[:, 7] = (d.y[:, 7] - mean) / std
    
    torch.save({"mean": mean, "std": std}, "energy_norm.pt")

    return data

def getQM9data():
    # Data
    import torch
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split

    # dataset = QM9(root="Data/QM9")
    # dataset = dataset.shuffle()

    # subdataset = []
    # for i, data in enumerate(dataset):
    #     if data.num_nodes <= 10:
    #         subdataset.append(data)

    # subdataset = [data for data in dataset if data.z.size(0) <= 10]
    # print('subdataset: ', len(subdataset))
    # torch.save(subdataset, './Data/QM9/processed/miniQM9.pt')
    # exit()

    subdataset = MiniQM9(root="Data/QM9")
    print('subdataset[0]-y: ', subdataset[0].y.shape)

    # subdataset = normalize(subdataset)
    print('subdataset: ', len(subdataset))
    # print('y: ', subdataset[0].y[:, 7])

    traindata, tempdata = train_test_split(subdataset, test_size=0.2, train_size=0.8, random_state=42)
    testdata, valdata = train_test_split(tempdata, test_size=0.5, train_size=0.5, random_state=42)

    trainloader = DataLoader(traindata, batch_size=32, shuffle=True)
    valloader = DataLoader(valdata, batch_size=32)
    testloader = DataLoader(testdata, batch_size=32)
    print('train size: ', len(traindata))
    print('test size: ', len(testdata))
    print('val size: ', len(valdata))

    return trainloader, valloader, testloader


def getQM7bdata():
    import deepchem as dc

    # Load QM7 dataset
    tasks, datasets, transformers = dc.molnet.load_qm7(featurizer='GraphConv')
    train, valid, test = datasets

    print(len(train), len(valid), len(test))


# getQM7bdata()
# trainloader, _, _ = getQM9data()

# for t in trainloader:
#     print('U0: ', t.y[:,7])
#     exit()