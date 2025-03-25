import torch
import torch.nn.functional as F


def triplet_loss(embeddings, labels, margin=0.2):
    """
    Vectorized semi-hard triplet loss.
    Args:
        embeddings: (B, D) normalized embeddings
        labels: (B,) pseudo-labels (int64)
        margin: float, margin for triplet loss
    Returns:
        scalar tensor
    """
    device = embeddings.device
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)

    labels = labels.unsqueeze(1)  # (B, 1)
    matches = labels == labels.T  # (B, B) bool mask

    # Remove diagonal (self-matching)
    mask_pos = matches & ~torch.eye(matches.size(0), dtype=torch.bool, device=device)
    mask_neg = ~matches

    # For each anchor, get hardest positive
    pos_sim = sim_matrix.masked_fill(~mask_pos, float('-inf')).max(dim=1).values  # (B,)

    # For each anchor, get semi-hard negatives
    neg_sim = sim_matrix.masked_fill(~mask_neg, float('-inf'))  # (B, B)
    semi_hard_neg_mask = neg_sim > (pos_sim.unsqueeze(1) - margin)
    semi_hard_neg = neg_sim.masked_fill(~semi_hard_neg_mask, float('inf'))

    # Select minimum semi-hard negative sim per anchor
    neg_sim_min = semi_hard_neg.min(dim=1).values  # (B,)
    valid = ~torch.isinf(neg_sim_min)

    # Compute triplet loss
    loss = F.relu(pos_sim[valid] - neg_sim_min[valid] + margin)
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=device)



def classifier_loss(z_i, z_j):
    """
    Paper-consistent classifier loss:
    For each anchor, the positive is its paired view.
    """
    z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    # z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)  # (2B, D)
    sim_matrix = torch.matmul(z, z.T)                     # (2B, 2B)

    N = z.size(0)
    sim_matrix.fill_diagonal_(-float('inf'))

    # Positives are at index i <-> i ± B
    targets = torch.arange(N, device=z.device)
    targets = targets + N//2
    targets = targets % N  # maps i <-> i ± B

    return F.cross_entropy(sim_matrix, targets)




