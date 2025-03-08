import torch
import torch.nn as nn
import torch.nn.functional as F

def ntxent_loss(z_i, z_j, cfg):
    """
    NTXent Loss function.
    Parameters
    ----------
    z_i : torch.tensor
        embedding of original samples (batch_size x emb_size)
    z_j : torch.tensor
        embedding of augmented samples (batch_size x emb_size)
    Returns
    -------
    loss
    """
    tau = cfg['tau']
    z = torch.stack((z_i,z_j), dim=1).view(2*z_i.shape[0], z_i.shape[1])
    a = torch.matmul(z, z.T)
    a /= tau
    Ls = []
    for i in range(z.shape[0]):
        nn_self = torch.cat([a[i,:i], a[i,i+1:]])
        softmax = F.log_softmax(nn_self, dim=0)
        Ls.append(softmax[i if i%2 == 0 else i-1])
    Ls = torch.stack(Ls)
    
    loss = torch.sum(Ls) / -z.shape[0]
    return loss

def moco_loss(z_i, z_j, k_i, k_j, queue, cfg):
    tau = cfg['tau']
    batch_size = z_i.shape[0]

    # Compute logits
    positives = torch.cat([torch.einsum('nc,nc->n', [z_i, k_i]), torch.einsum('nc,nc->n', [z_j, k_j])], dim=0).unsqueeze(-1)
    negatives = torch.cat([torch.einsum('nc,ck->nk', [z_i, queue]), torch.einsum('nc,ck->nk', [z_j, queue])], dim=0)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= tau

    # Labels: positives are at index 0
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


# For mixup-based contrastive learning
class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, logits, target):
        probs = F.softmax(logits, 1) 
        loss = (- target * torch.log(probs + 1e-9)).sum(1).mean()

        return loss