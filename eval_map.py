import json
import numpy as np
from collections import defaultdict
import faiss

from eval import load_memmap_data, get_index, extract_test_ids


def calculate_map(ground_truth, predictions, test_seq_len, k=10):
    """
    Computes the Mean Average Precision (MAP) separately for each query length.
    Returns:
    - Dictionary mapping query lengths to MAP scores.
    - Overall weighted MAP.
    """
    map_per_length = defaultdict(list)
    
    print(f"Total queries in predictions: {len(predictions)}")
    print(f"Total queries in ground_truth: {len(ground_truth)}")
    
    for q_id, retrieved_list in predictions.items():
        if q_id not in ground_truth:
            print(f"Skipping {q_id} - not found in ground truth")
            continue  # Skip if no ground truth exists

        relevant_items = ground_truth[q_id]
        num_relevant = 0
        precision_values = []

        for i, retrieved_id in enumerate(retrieved_list[:k]):
            if retrieved_id in relevant_items:
                num_relevant += 1
                precision_values.append(num_relevant / (i + 1))  # Precision@i
        
        if not precision_values:
            print(f"No relevant items found for query {q_id} in top-{k} retrieved list")

        ap = np.mean(precision_values) if precision_values else 0
        query_length = len(retrieved_list)  # Approximate segment length

        print(f"Query {q_id} | Length {query_length} | AP: {ap}")
        map_per_length[query_length].append(ap)

    if not map_per_length:
        print("ERROR: No valid MAP calculations were made!")

    # Compute MAP for each length
    map_per_length = {length: np.mean(scores) for length, scores in map_per_length.items()}

    # Compute weighted MAP
    total_queries = sum(len(scores) for scores in map_per_length.values())
    weighted_map = sum(np.mean(scores) * len(scores) / total_queries for length, scores in map_per_length.items()) if total_queries > 0 else 0

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

    query, query_shape = load_memmap_data(emb_dir, 'query_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir or emb_dir, 'dummy_db')

    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                      max_train, n_centroids=n_centroids)
    
    index.add(dummy_db)
    index.add(db)

    # Load lookup tables
    query_lookup = json.load(open(f'{emb_dir}/query_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))

    with open('data/gt_dict.json', 'r') as fp:
        ground_truth = json.load(fp)

    test_ids, max_test_seq_len = extract_test_ids(query_lookup)
    predictions = {}

    for ti, test_id in enumerate(test_ids):
        max_len = int(max_test_seq_len[ti])
        max_query_len = np.array(list(map(int, test_seq_len.split())))
        max_query_len = max_query_len[max_query_len <= max_len]

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

            sorted_predictions = sorted(hist, key=hist.get, reverse=True)
            predictions[q_id] = sorted_predictions

    # Compute MAP
    map_score = calculate_map(ground_truth, predictions, k=k_map)

    print(f'MAP@{k_map}: {map_score:.4f}')

    return map_score, k_map
