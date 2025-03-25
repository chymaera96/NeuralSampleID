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
    Vectorized contrastive classification loss with proper masking.
    Args:
        embeddings: (2B, D) normalized
        labels: (2B,) where positives have same label
    Returns:
        scalar loss (mean over samples)
    """
    device = embeddings.device
    N = labels.size(0)
    
    # Cosine similarity matrix
    sim = torch.matmul(embeddings, embeddings.T)  # shape: (N, N)
    
    # Mask self-similarity (avoid log(0) in softmax)
    sim.fill_diagonal_(-float('inf'))

    # Build mask of positives (excluding self)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~torch.eye(N, device=device, dtype=torch.bool))

    # Compute log-softmax over rows
    log_probs = F.log_softmax(sim, dim=1)

    # Sanity checks
    pos_per_row = pos_mask.sum(dim=1)
    if (pos_per_row == 0).any():
        print("❌ NaN risk: some rows have 0 positives!")
        print("pos_mask.sum(1):", pos_per_row.tolist())

    # Compute loss per sample, handle divide-by-zero via clamp
    loss = - (log_probs * pos_mask.float()).sum(dim=1) / pos_per_row.clamp(min=1)

    # Final safety check
    if torch.isnan(loss).any():
        print("🧨 NaN in classifier loss!")
        print("sim min/max:", sim.min().item(), sim.max().item())
        print("log_probs min/max:", log_probs.min().item(), log_probs.max().item())
        print("loss vector:", loss)

        return torch.tensor(0.0, device=device)

    return loss.mean()



