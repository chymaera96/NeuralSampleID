import json
import numpy as np
from collections import defaultdict
import faiss

from eval import load_memmap_data, get_index, extract_test_ids


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




def eval_faiss_with_map(emb_dir,
                         emb_dummy_dir=None,
                         index_type='ivfpq',
                         nogpu=False,
                         max_train=1e7,
                         test_ids='icassp',
                         test_seq_len='1 3 5 9 11 19',
                         k_probe=5,
                         n_centroids=64,
                         k_map=10):
    """
    Extended evaluation function to compute Mean Average Precision (MAP).
    """

    if type(test_seq_len) == str:
        test_seq_len = np.asarray(
            list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]
    elif type(test_seq_len) == list:
        test_seq_len = np.asarray(test_seq_len)

    query, query_shape = load_memmap_data(emb_dir, 'query_full_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')

    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                      max_train, n_centroids=n_centroids)

    index.add(dummy_db)
    index.add(db)
    del dummy_db

    fake_recon_index, index_shape = load_memmap_data(
        emb_dummy_dir, 'dummy_db', append_extra_length=db_shape[0],
        display=False)
    fake_recon_index[dummy_db_shape[0]:dummy_db_shape[0] + db_shape[0], :] = db[:, :]
    fake_recon_index.flush()

    # Load lookup tables
    query_lookup = json.load(open(f'{emb_dir}/query_full_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))

    with open('data/gt_dict.json', 'r') as fp:
        ground_truth = json.load(fp)

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
            if cid < dummy_db_shape[0]:
                continue
            match = ref_lookup[cid - dummy_db_shape[0]]
            if match == q_id:
                continue
            candidate_seq = fake_recon_index[cid:(cid + q.shape[0]), :]
            if candidate_seq.shape[0] < q.shape[0]:
                q_match = q[:candidate_seq.shape[0], :]
            else:
                q_match = q
            score = np.mean(np.sum(q_match * candidate_seq, axis=1))
            hist[match] += score
        
        if ix % 20 == 0:
            print(f"Processed {ix} / {len(test_ids)} queries...")

        sorted_predictions = sorted(hist, key=hist.get, reverse=True)
        predictions[q_id] = sorted_predictions

    # Compute MAP
    map_score = calculate_map(ground_truth, predictions, k=k_map)

    del query, db, fake_recon_index

    np.save(f'{emb_dir}/predictions.npy', predictions)
    np.save(f'{emb_dir}/map_score.npy', map_score)
    print(f"Saved predictions and MAP score to {emb_dir}")

    return map_score, k_map
