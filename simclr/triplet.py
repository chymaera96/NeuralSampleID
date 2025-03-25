import torch
import torch.nn.functional as F


def get_batch_labels(batch_size, device):
    """
    Generate pseudo-labels for a batch of (x_i, x_j) pairs.
    Returns:
        labels: Tensor of shape (2 * batch_size,)
    """
    labels = torch.arange(batch_size, device=device)
    return torch.cat([labels, labels], dim=0)


def triplet_loss(embeddings, labels, margin=0.2):
    """
    Semi-hard triplet loss with in-batch mining.
    Args:
        embeddings: (2B, D) normalized embeddings
        labels: (2B,) pseudo-labels
        margin: Margin for triplet loss
    Returns:
        loss: Scalar loss value
    """
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # cosine similarity
    batch_size = labels.size(0)
    loss = 0.0
    count = 0

    for i in range(batch_size):
        anchor_label = labels[i]
        anchor_sim = sim_matrix[i]

        # Positive and negative masks
        is_pos = (labels == anchor_label) & (torch.arange(batch_size, device=labels.device) != i)
        is_neg = (labels != anchor_label)

        if not torch.any(is_pos) or not torch.any(is_neg):
            continue

        pos_sim = anchor_sim[is_pos]         # (P,)
        neg_sim = anchor_sim[is_neg]         # (N,)

        pos_sim = pos_sim.max()              # hardest positive
        semi_hard_negs = neg_sim[neg_sim > pos_sim - margin]

        if semi_hard_negs.numel() == 0:
            continue

        neg_sim = semi_hard_negs.min()       # hardest semi-hard negative

        triplet = F.relu(pos_sim - neg_sim + margin)
        loss += triplet
        count += 1

    return loss / max(count, 1)


def classifier_loss(embeddings, labels):
    """
    Contrastive classification loss using dot-product + softmax.
    Args:
        embeddings: (2B, D) normalized embeddings
        labels: (2B,) pseudo-labels (same label => same class)
    """
    logits = torch.matmul(embeddings, embeddings.T)
    logits = logits - torch.eye(logits.size(0), device=logits.device) * 1e12  # mask diagonal

    return F.cross_entropy(logits, labels)
