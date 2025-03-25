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



def classifier_loss(embeddings, labels):
    """
    Contrastive classification loss using dot-product + softmax.
    Args:
        embeddings: (2B, D) normalized embeddings
        labels: (2B,) pseudo-labels (same label => same class)
    """
    logits = torch.matmul(embeddings, embeddings.T)
    logits.fill_diagonal_(-float('inf'))

    return F.cross_entropy(logits, labels)
