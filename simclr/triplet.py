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
    Adapted NT-Xent contrastive loss for classification-style pairing.
    Each (z_i[k], z_j[k]) is a positive pair.
    """
    z = torch.cat([z_i, z_j], dim=0)              # (2B, D)
    sim_matrix = torch.matmul(z, z.T)

    N = z.shape[0]
    labels = torch.arange(N // 2, device=z.device)
    labels = torch.cat([labels, labels], dim=0)   # [0, 1, ..., B-1, 0, 1, ..., B-1]

    loss = 0
    for i in range(N):
        # mask out self
        sim_i = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        label_i = labels[i]
        labels_wo_self = torch.cat([labels[:i], labels[i+1:]])

        # identify positives (same class)
        pos_mask = (labels_wo_self == label_i)

        if pos_mask.sum() == 0:
            continue  # no positives

        log_prob = F.log_softmax(sim_i, dim=0)
        loss += -log_prob[pos_mask].mean()

    return loss / N



