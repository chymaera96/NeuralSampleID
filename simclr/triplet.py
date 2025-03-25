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
    Vectorized supervised contrastive loss (no temperature, no loop).
    Args:
        embeddings: (2B, D) normalized
        labels: (2B,) with matching labels for positives
    """
    device = embeddings.device
    sim = torch.matmul(embeddings, embeddings.T)  # cosine sim (2B, 2B)
    B = labels.size(0)

    # Remove self-similarity from numerator and denominator
    mask = torch.eye(B, dtype=torch.bool, device=device)
    sim.masked_fill_(mask, -float('inf'))

    # Create binary mask of positive pairs (excluding self)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    positive_mask = label_eq & ~mask  # exclude self

    # Compute log-softmax across rows
    log_probs = F.log_softmax(sim, dim=1)  # (B, B)

    # Only keep log-probs of actual positives
    loss = - (log_probs * positive_mask.float()).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)

    return loss.mean()

