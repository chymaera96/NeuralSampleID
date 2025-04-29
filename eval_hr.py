import torch
import faiss
import time
import numpy as np
import os
from collections import defaultdict
import json

from eval import load_memmap_data, get_index

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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




def eval_faiss_clf(emb_dir,
                   classifier,
                   emb_dummy_dir=None,
                   index_type='ivfpq',
                   nogpu=False,
                   max_train=1e7,
                   test_ids='icassp',
                   test_seq_len='1 3 5 9 11 19',
                   k_probe=5,
                   n_centroids=64):


    classifier.to(device).eval()

    query_nmatrix_path = os.path.join(emb_dir, 'query_nmatrix.npy')
    ref_nmatrix_dir = os.path.join(emb_dir, 'ref_nmatrix')

    if isinstance(test_seq_len, str):
        test_seq_len = np.array(list(map(int, test_seq_len.split())))
    elif isinstance(test_seq_len, list):
        test_seq_len = np.array(test_seq_len)

    query_lookup = json.load(open(f'{emb_dir}/query_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))
    query_nmatrix = np.load(query_nmatrix_path, allow_pickle=True).item()
    with open('data/gt_dict.json', 'r') as fp:
        gt = json.load(fp)

    query, _ = load_memmap_data(emb_dir, 'query_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir or emb_dir, 'dummy_db')

    index = get_index(index_type, dummy_db, dummy_db.shape, not nogpu, max_train, n_centroids)
    index.add(dummy_db)
    index.add(db)

    test_ids, max_test_seq_len = extract_test_ids(query_lookup)
    ref_song_starts, _ = extract_test_ids(ref_lookup)
    n_test = len(test_ids)

    top1, top3, top10 = np.zeros((n_test, len(test_seq_len)), dtype=int), \
                        np.zeros((n_test, len(test_seq_len)), dtype=int), \
                        np.zeros((n_test, len(test_seq_len)), dtype=int)

    for ti, test_id in enumerate(test_ids):
        q_name = query_lookup[test_id]
        q_id = q_name.split('_')[0]
        max_len = int(max_test_seq_len[ti])
        max_query_len = test_seq_len[test_seq_len <= max_len]
        nm_query_full = torch.tensor(query_nmatrix[q_id]).to(device)

        for si, sl in enumerate(max_query_len):
            # print(f"-----------Processing {q_name} with {sl} segments-----------")
            q = query[test_id:(test_id + sl), :]
            # print(f"nm_query_full shape: {nm_query_full.shape}")
            nm_query = nm_query_full[:sl, :, :]
            # print(f"nm_query shape: {nm_query.shape}")

            _, I = index.search(q, k_probe)
            candidates = I[np.where(I >= 0)].flatten()

            hist = defaultdict(float)

            for ix,cid in enumerate(candidates):
                # print(f"PROCESSING {ix}/{len(candidates)}")
                if cid < dummy_db_shape[0]:
                    continue
                ref_id = cid - dummy_db_shape[0]
                match = ref_lookup[ref_id]
                if match == q_id:
                    continue

                song_start_idx = ref_song_starts[ref_song_starts <= ref_id].max()
                segment_idx = ref_id - song_start_idx

                ref_path = os.path.join(ref_nmatrix_dir, f"{match}.npy")
                if not os.path.exists(ref_path):
                    print(f"Missing reference matrix for {match}, skipping...")
                    continue
                ref_nmat = np.load(ref_path)
                if segment_idx >= ref_nmat.shape[0]:
                    print(f"Segment index {segment_idx} out of bounds for {match}, skipping...")
                    continue

                nm_candidate = torch.tensor(ref_nmat[segment_idx]).to(device)
                nm_candidate = nm_candidate.unsqueeze(0).repeat(nm_query.shape[0], 1, 1)  # Match query shape
                # print(f"nm_candidate shape: {nm_candidate.shape}")

                with torch.no_grad():
                    logits = classifier(nm_query, nm_candidate)
                    score = logits.max().item()
            

                if score >= 0.5:
                    hist[match] += score

                # print(f"Classifier score for {match}: {score:.4f} (before freq weighting)")

            pred = sorted(hist, key=hist.get, reverse=True)

            if pred:
                top1[ti, si] = int(q_id in gt[pred[0]])
                top3[ti, si] = int(any(q_id in gt[p] for p in pred[:3]))
                top10[ti, si] = int(any(q_id in gt[p] for p in pred[:10]))
        
    log_interval = 100 if len(test_ids) > 500 else 10

    if ti % log_interval == 0:
        print(f"Processed {ti} / {len(test_ids)} queries...")

    valid = (test_seq_len <= max_test_seq_len[:, None])
    hit_rates = np.stack([
        100 * np.nanmean(np.where(valid, top1, np.nan), axis=0),
        100 * np.nanmean(np.where(valid, top3, np.nan), axis=0),
        100 * np.nanmean(np.where(valid, top10, np.nan), axis=0),
    ], axis=0)

    np.save(f'{emb_dir}/hit_rates_clf.npy', hit_rates)
    np.save(f'{emb_dir}/raw_score_clf.npy', np.concatenate([top1, top3, top10], axis=1))
    np.save(f'{emb_dir}/test_ids_clf.npy', test_ids)
    print(f"Saved classifier-based hit-rates to {emb_dir}")

    return hit_rates
