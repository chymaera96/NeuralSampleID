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





def eval_faiss_map_clf(emb_dir, classifier, emb_dummy_dir=None,
                       index_type='ivfpq', nogpu=False, max_train=1e7,
                       k_probe=3, n_centroids=32, k_map=20):
    """
    Evaluation using classifier logits instead of cosine similarity.
    """
    classifier.to(device).eval()

    query_nmatrix_path = os.path.join(emb_dir, 'query_full_nmatrix.npy')
    ref_nmatrix_dir = os.path.join(emb_dir, 'ref_nmatrix')

    # Load FAISS index
    query, query_shape = load_memmap_data(emb_dir, 'query_full_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')

    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu), max_train, n_centroids=n_centroids)
    index.add(dummy_db)
    index.add(db)
    del dummy_db

    # Load lookup tables
    query_lookup = json.load(open(f'{emb_dir}/query_full_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))

    with open('data/gt_dict.json', 'r') as fp:
        ground_truth = json.load(fp)

    # Load query node matrices
    query_nmatrix = np.load(query_nmatrix_path, allow_pickle=True).item()
    test_ids, max_test_seq_len = extract_test_ids(query_lookup)
    ref_song_starts, _ = extract_test_ids(ref_lookup)
    
    predictions = {}

    print("Starting FAISS-based retrieval and ranking...")

    for ix, test_id in enumerate(test_ids):
        q_id = query_lookup[test_id].split("_")[0]
        max_len = max_test_seq_len[ix]
        q = query[test_id: test_id + max_len, :]

        _, I = index.search(q, k_probe)

        candidates, freqs = np.unique(I[I >= 0], return_counts=True)
        # print(f"\nQuery {ix}: Retrieved {len(candidates)} candidates.")

        hist = defaultdict(int)
        for cid, freq in zip(candidates, freqs):
            if cid < dummy_db_shape[0]:
                continue
            cid = cid - dummy_db_shape[0]
            match = ref_lookup[cid]
            if match == q_id:
                continue

            # Get correct segment index in the retrieved song
            song_start_idx = ref_song_starts[ref_song_starts <= cid].max()
            ref_seg_idx = cid - song_start_idx

            # Load the reference node matrix
            ref_nmatrix_path = os.path.join(ref_nmatrix_dir, f"{match}.npy")
            if not os.path.exists(ref_nmatrix_path):
                print(f"Missing reference matrix for {match}, skipping...")
                continue

            ref_nmatrix = np.load(ref_nmatrix_path)  # (num_segments, C, N)
            if ref_seg_idx >= ref_nmatrix.shape[0]:
                print(f"Segment index {ref_seg_idx} out of bounds for {match}, skipping...")
                continue  

            nm_candidate = torch.tensor(ref_nmatrix[ref_seg_idx]).to(device)
            nm_query = torch.tensor(query_nmatrix[q_id]).to(device)

            # Ensure nm_candidate is repeated across nm_query's segments
            nm_candidate = nm_candidate.unsqueeze(0).repeat(nm_query.shape[0], 1, 1)  # (num_segments, C, N)

            # Compute classifier logits in batch mode
            logits = classifier(nm_query, nm_candidate)  # (num_segments, 1)

            clf_score = logits.max().item()

            # weighted_score = clf_score * np.log1p(freq) if clf_score > 0.5 else 0
            weighted_score = clf_score if clf_score > 0.5 else 0
            hist[match] += weighted_score

        # print(f"Top 10 scores for {q_id}: {sorted(hist.items(), key=lambda x: x[1], reverse=True)[:10]}")
        if ix % 5 == 0:
            print(f"Processed {ix} / {len(test_ids)} queries...")

        predictions[q_id] = sorted(hist, key=hist.get, reverse=True)

    # Compute MAP
    print("\nComputing MAP score...")
    map_score = calculate_map(ground_truth, predictions, k=k_map)
    np.save(f'{emb_dir}/predictions.npy', predictions)
    np.save(f'{emb_dir}/map_score.npy', map_score)
    
    print(f"MAP score computed: {map_score:.4f}")
    print(f"Saved predictions and MAP score to {emb_dir}")

    return map_score, k_map

