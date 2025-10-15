import torch
import torch.nn as nn

from gaussian import gaussian_expand_torch


class ElementWiseProduct(nn.Module):
    def __init__(self):
        super().__init__()

    

class DeepTensorNN(nn.Module):
    def __init__(self, numatoms, dfeatdim, outputdim):
        super(DeepTensorNN, self).__init__()
        self.numatoms = numatoms
        self.atomemb_dim = 20
        
        self.atomdesc = nn.Embedding(numatoms, self.atomemb_dim)
        
        self.V = nn.Sequential(
            nn.Linear(self.atomemb_dim + dfeatdim, self.atomemb_dim),
            nn.Tanh()
        )

        self.top = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.atomemb_dim, 10),
            nn.Linear(10, outputdim)
        )

    def forward(self, z: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        mask = z != 0

        cfeat = self.atomdesc(z)

        cfeat = cfeat * mask.unsqueeze(2).expand(-1, -1, self.atomemb_dim)
        
        cfeat_expand = cfeat.clone().unsqueeze(2).expand(-1, -1, self.numatoms, -1)
        cfeat_expand = cfeat_expand * mask.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.numatoms, self.atomemb_dim)
        
        mu_min, mu_max, delta_mu, sigma = 0.0, 5.0, 0.2, 0.5
        dfeat = gaussian_expand_torch(dist, mu_min, mu_max, delta_mu, sigma)
        
        pairfeats = torch.cat([cfeat_expand, dfeat], dim=-1)
        
        message_ij = self.V(pairfeats)
        message_ij = message_ij * mask.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.numatoms, self.atomemb_dim)

        aggmsg = message_ij.sum(dim=2)

        cfeat = cfeat + aggmsg
        
        energies = self.top(cfeat)
        
        energy = energies.sum(dim=1).squeeze()

        return energy
    

