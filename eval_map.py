import json
import numpy as np
from collections import defaultdict
import faiss

from eval import load_memmap_data, get_index, extract_test_ids


def calculate_map(ground_truth, predictions, test_seq_len, k=10):
    """
    Computes the Mean Average Precision (MAP) separately for each query length.

    Parameters:
    - ground_truth: Dictionary mapping query IDs to sets of correct matches.
    - predictions: Dictionary where each query ID maps to its list of retrieved tracks.
    - test_seq_len: List or array of query segment lengths.
    - k: Number of top results to consider.

    Returns:
    - Dictionary mapping query lengths to MAP scores.
    - Overall weighted MAP.
    """
    map_per_length = defaultdict(list)
    
    for q_id, retrieved_list in predictions.items():
        if q_id not in ground_truth:
            continue  # Skip if no ground truth exists

        relevant_items = ground_truth[q_id]
        num_relevant = 0
        precision_values = []

        for i, retrieved_id in enumerate(retrieved_list[:k]):
            if retrieved_id in relevant_items:
                num_relevant += 1
                precision_values.append(num_relevant / (i + 1))  # Precision@i

        ap = np.mean(precision_values) if precision_values else 0
        query_length = len(retrieved_list)  # Approximate segment length
        map_per_length[query_length].append(ap)

    # Compute MAP for each length
    map_per_length = {length: np.mean(scores) for length, scores in map_per_length.items()}

    # Compute weighted MAP
    total_queries = sum(len(scores) for scores in map_per_length.values())
    weighted_map = sum(np.mean(scores) * len(scores) / total_queries for length, scores in map_per_length.items())

    return map_per_length, weighted_map



def eval_faiss_with_map(emb_dir,
                         emb_dummy_dir=None,
                         index_type='ivfpq',
                         nogpu=False,
                         max_train=1e7,
                         test_ids='icassp',
                         test_seq_len='1 3 5 9 11 19',
                         k_probe=20,
                         n_centroids=64,
                         k_map=20):
    """
    Extended evaluation function to compute Mean Average Precision (MAP).
    """
    if type(test_seq_len) == str:
        test_seq_len = np.asarray(
            list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]
    elif type(test_seq_len) == list:
        test_seq_len = np.asarray(test_seq_len)

    query, query_shape = load_memmap_data(emb_dir, 'query_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir or emb_dir, 'dummy_db')
    
    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                      max_train, n_centroids=n_centroids)
    
    index.add(dummy_db)
    index.add(db)

    print(f'test_id: \033[93m{test_ids}\033[0m,  ', end='')
    
    # Load lookup tables
    query_lookup = json.load(open(f'{emb_dir}/query_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))
    
    with open('data/gt_dict.json', 'r') as fp:
        ground_truth = json.load(fp)
    
    test_ids, max_test_seq_len = extract_test_ids(query_lookup)
    print(f'n_test: \033[93m{len(test_ids):n}\033[0m')

    predictions = {}
    
    for ti, test_id in enumerate(test_ids):
        max_len = int(max_test_seq_len[ti])
        max_query_len = test_seq_len[test_seq_len <= max_len]
        
        for sl in max_query_len:
            q = query[test_id:(test_id + sl), :]
            q_id = query_lookup[test_id].split("_")[0]
            _, I = index.search(q, k_probe)
            candidates = I[np.where(I >= 0)].flatten()
            
            hist = defaultdict(int)
            
            for cid in candidates:
                if cid < dummy_db_shape[0]:
                    continue
                match = ref_lookup[cid - dummy_db_shape[0]]
                if match == q_id:
                    continue
                hist[match] += 1
            
            predictions[q_id] = sorted(hist, key=hist.get, reverse=True)
            
    print(predictions)
    map_per_length, weighted_map = calculate_map(ground_truth, predictions, test_seq_len, k=k_map)

    print(f"MAP per query length: {map_per_length}")
    print(f"Weighted MAP: {weighted_map:.4f}")

    return weighted_map, k_map
