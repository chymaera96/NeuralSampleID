import os
import json
import torch
import numpy as np
from collections import defaultdict
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from eval import load_memmap_data, get_index

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def calculate_map(ground_truth, predictions, k=10):
    """
    Computes the Mean Average Precision (MAP) at k.

    Parameters:
    - ground_truth: Dictionary mapping query IDs to correct matches.
    - predictions: Dictionary where each query ID maps to its list of retrieved tracks.
    - k: Number of top results to consider.

    Returns:
    - MAP score.
    """
    average_precisions = []

    for q_id, retrieved_list in predictions.items():
        num_relevant = 0
        precision_values = []

        for i, retrieved_id in enumerate(retrieved_list[:k]):
            if q_id in ground_truth.get(retrieved_id, []):
                num_relevant += 1
                precision_values.append(num_relevant / (i + 1))  # Precision@i

        ap = np.mean(precision_values) if precision_values else 0
        average_precisions.append(ap)

    return np.mean(average_precisions) if average_precisions else 0



def extract_test_ids(lookup_table):
    starts = []
    lengths = []
    
    # Initialize the first string and starting index
    current_string = lookup_table[0]
    current_start = 0
    
    # Iterate through the list to detect changes in strings
    for i in range(1, len(lookup_table)):
        if lookup_table[i] != current_string:
            # When a new string is found, record the start and length of the previous group
            starts.append(current_start)
            lengths.append(i - current_start)
            
            # Update the current string and starting index
            current_string = lookup_table[i]
            current_start = i
    
    # Add the last group
    starts.append(current_start)
    lengths.append(len(lookup_table) - current_start)
    
    return np.array(starts), np.array(lengths)





def eval_faiss_with_map_classifier(emb_dir, classifier, 
                                   index_type='ivfpq', nogpu=False, max_train=1e7,
                                   k_probe=20, n_centroids=32, k_map=20):
    """
    Evaluation using classifier logits instead of cosine similarity.
    """
    classifier.to(device).eval()

    query_nmatrix_path = os.path.join(emb_dir, 'query_nmatrix.npy')
    ref_nmatrix_dir = os.path.join(emb_dir, 'ref_nmatrix')

    # Load FAISS index
    query, query_shape = load_memmap_data(emb_dir, 'query_full_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    index = get_index(index_type, db, db.shape, (not nogpu), max_train, n_centroids=n_centroids)
    index.add(db)
    del db

    # Load lookup tables
    query_lookup = json.load(open(f'{emb_dir}/query_full_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))

    with open('data/gt_dict.json', 'r') as fp:
        ground_truth = json.load(fp)

    # Load query node matrices
    query_nmatrix = np.load(query_nmatrix_path, allow_pickle=True).item()
    test_ids, max_test_seq_len = extract_test_ids(query_lookup)
    predictions = {}

    for ix, test_id in enumerate(test_ids):
        q_id = query_lookup[test_id].split("_")[0]
        max_len = max_test_seq_len[ix]
        q = query[test_id: test_id + max_len, :]
        if q.shape[0] <= 10:
            continue

        _, I = index.search(q, k_probe)
        candidates = I[np.where(I >= 0)].flatten()

        hist = defaultdict(int)
        for cid in candidates:
            if cid < db_shape[0]:
                continue
            match = ref_lookup[cid - db_shape[0]]
            if match == q_id:
                continue

            # Get correct segment index in the retrieved song
            ref_song_starts, _ = extract_test_ids(ref_lookup)
            song_start_idx = ref_song_starts[ref_song_starts <= cid].max()
            segment_idx = cid - song_start_idx

            # Load the reference node matrix
            ref_nmatrix_path = os.path.join(ref_nmatrix_dir, f"{match}.npy")
            if not os.path.exists(ref_nmatrix_path):
                continue  # Skip if missing reference

            ref_nmatrix = np.load(ref_nmatrix_path)  # (num_segments, C, N)
            if segment_idx >= ref_nmatrix.shape[0]:
                continue  # Skip if index out of bounds

            x_before_proj_candidate = torch.tensor(ref_nmatrix[segment_idx]).to(device)
            x_before_proj_query = torch.tensor(query_nmatrix[q_id]).to(device)

            # Compute classifier score
            classifier_score = classifier(x_before_proj_query, x_before_proj_candidate).item()
            hist[match] += classifier_score

        sorted_predictions = sorted(hist, key=hist.get, reverse=True)
        predictions[q_id] = sorted_predictions

        if ix % 20 == 0:
            print(f"Processed {ix} / {len(test_ids)} queries...")

    # Compute MAP
    map_score = calculate_map(ground_truth, predictions, k=k_map)
    np.save(f'{emb_dir}/predictions.npy', predictions)
    np.save(f'{emb_dir}/map_score.npy', map_score)
    print(f"Saved predictions and MAP score to {emb_dir}")

    return map_score, k_map

