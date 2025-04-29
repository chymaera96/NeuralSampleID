import faiss
import time
import numpy as np
import os
from collections import defaultdict
import json


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
    
    starts.append(current_start)
    lengths.append(len(lookup_table) - current_start)
    
    return np.array(starts), np.array(lengths)


def get_index(index_type,
              train_data,
              train_data_shape,
              use_gpu=True,
              max_nitem_train=2e7,
              n_centroids=64,
):
    """
    • Create FAISS index
    • Train index using (partial) data
    • Return index
    Parameters
    ----------
    index_type : (str)
        Index type must be one of {'L2', 'IVF', 'IVFPQ', 'IVFPQ-RR',
                                   'IVFPQ-ONDISK', HNSW'}
    train_data : (float32)
        numpy.memmap or numpy.ndarray
    train_data_shape : list(int, int)
        Data shape (n, d). n is the number of items. d is dimension.
    use_gpu: (bool)
        If False, use CPU. Default is True.
    max_nitem_train : (int)
        Max number of items to be used for training index. Default is 1e7.
    Returns
    -------
    index : (faiss.swigfaiss_avx2.GpuIndex***)
        Trained FAISS index.
    References:
        https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """
    # GPU Setup
    if use_gpu:
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_OPTIONS = faiss.GpuClonerOptions()
        GPU_OPTIONS.useFloat16 = True # use float16 table to avoid https://github.com/facebookresearch/faiss/issues/1178
        #GPU_OPTIONS.usePrecomputed = False
        #GPU_OPTIONS.indicesOptions = faiss.INDICES_CPU
    else:
        pass

    # Fingerprint dimension, d
    d = train_data_shape[1]

    # Build a flat (CPU) index
    index = faiss.IndexFlatL2(d) #

    mode = index_type.lower()
    print(f'Creating index: \033[93m{mode}\033[0m')
    if mode == 'l2':
        # Using L2 index
        pass
    elif mode == 'ivf':
        # Using IVF index
        nlist = 400
        index = faiss.IndexIVFFlat(index, d, nlist)
    elif mode == 'ivfpq':
        # Using IVF-PQ index
        code_sz = 64 # power of 2
        nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)

    elif mode == 'lsh':
        # Using LSH index
        nbits = 256
        index = faiss.IndexLSH(d, nbits)


    elif mode == 'ivfpq-rr':
        # Using IVF-PQ index + Re-rank
        code_sz = 64
        # n_centroids = 256 # 10:1.92ms, 30:1.29ms, 100: 0.625ms
        nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
        M_refine = 4
        nbits_refine = 4
        index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits,
                                  M_refine, nbits_refine)
    elif mode == 'ivfpq-ondisk':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        raise NotImplementedError(mode)
    elif mode == 'hnsw':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        else:
            M = 16
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = 80
            index.verbose = True
            index.hnsw.search_bounded_queue = True
    else:
        raise ValueError(mode.lower())

    # From CPU index to GPU index
    if use_gpu:
        print('Copy index to \033[93mGPU\033[0m.')
        index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index, GPU_OPTIONS)

    # Train index
    start_time = time.time()
    if len(train_data) > max_nitem_train:
        print('Training index using {:>3.2f} % of data...'.format(
            100. * max_nitem_train / len(train_data)))
        # shuffle and reduce training data
        sel_tr_idx = np.random.permutation(len(train_data))
        sel_tr_idx = sel_tr_idx[:max_nitem_train]
        index.train(train_data[sel_tr_idx,:])
    else:
        print('Training index...')
        index.train(train_data) # Actually do nothing for {'l2', 'hnsw'}
    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_time))

    # N probe
    index.nprobe = 20
    return index


def load_memmap_data(source_dir,
                     fname,
                     append_extra_length=None,
                     shape_only=False,
                     display=True):
    """
    Load data and datashape from the file path.
    • Get shape from [source_dir/fname_shape.npy}.
    • Load memmap data from [source_dir/fname.mm].
    Parameters
    ----------
    source_dir : (str)
    fname : (str)
        File name except extension.
    append_empty_length : None or (int)
        Length to append empty vector when loading memmap. If activate, the
        file will be opened as 'r+' mode.
    shape_only : (bool), optional
        Return only shape. The default is False.
    display : (bool), optional
        The default is True.
    Returns
    -------
    (data, data_shape)
    """
    path_shape = os.path.join(source_dir, fname + '_shape.npy')
    path_data = os.path.join(source_dir, fname + '.mm')
    data_shape = np.load(path_shape)
    if shape_only:
        return data_shape

    if append_extra_length:
        data_shape[0] += append_extra_length
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    else:
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    # Convert nan values to 0
    data[np.isnan(data)] = 0.0
    if display:
        print(f'Load {data_shape[0]:,} items from \033[32m{path_data}\033[0m.')
    return data, data_shape

def eval_faiss(emb_dir,
               emb_dummy_dir=None,
               index_type='ivfpq',
               nogpu=False,
               max_train=1e7,
               test_ids='icassp',
               test_seq_len='1 3 5 9 11 19',
               k_probe=20,
               n_centroids=64):
    """
    Segment/sequence-wise audio search experiment and evaluation: implementation based on FAISS.
    """
    if type(test_seq_len) == str:
        test_seq_len = np.asarray(
            list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]
    elif type(test_seq_len) == list:
        test_seq_len = np.asarray(test_seq_len)
    # assert type(test_seq_len) == np.ndarray, f'{type(test_seq_len)} is not np.ndarray'
    
    # Load items from {query, db, dummy_db}
    query, query_shape = load_memmap_data(emb_dir, 'query_db')
    db, db_shape = load_memmap_data(emb_dir, 'ref_db')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')
    """ ----------------------------------------------------------------------
    FAISS index setup
        dummy: 10 items.
        db: 5 items.
        query: 5 items, corresponding to 'db'.
        index.add(dummy_db); index.add(db) # 'dummy_db' first
               |------ dummy_db ------|
        index: [d0, d1, d2,..., d8, d9, d11, d12, d13, d14, d15]
                                       |--------- db ----------|
                                       |--------query ---------|
                                       [q0,  q1,  q2,  q3,  q4]
    • The set of ground truth IDs for q[i] will be (i + len(dummy_db))
    ---------------------------------------------------------------------- """
    # Create and train FAISS index
    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                      max_train, n_centroids=n_centroids)

    # Add items to index
    start_time = time.time()

    index.add(dummy_db); print(f'{len(dummy_db)} items from dummy DB')
    index.add(db); print(f'{len(db)} items from reference DB')

    t = time.time() - start_time
    print(f'Added total {index.ntotal} items to DB. {t:>4.2f} sec.')

    """ ----------------------------------------------------------------------
    We need to prepare a merged {dummy_db + db} memmap:
    • Calcuation of sequence-level matching score requires reconstruction of
      vectors from FAISS index.
    • Unforunately, current faiss.index.reconstruct_n(id_start, id_stop)
      supports only CPU index.
    • We prepare a fake_recon_index thourgh the on-disk method.
    • In future implementations, fake_recon_index will be used for calculating
      histogram-based matching score.
    ---------------------------------------------------------------------- """
    # Prepare fake_recon_index
    del dummy_db
    start_time = time.time()

    fake_recon_index, index_shape = load_memmap_data(
        emb_dummy_dir, 'dummy_db', append_extra_length=db_shape[0],
        display=False)
    fake_recon_index[dummy_db_shape[0]:dummy_db_shape[0] + db_shape[0], :] = db[:, :]
    fake_recon_index.flush()

    t = time.time() - start_time
    print(f'Created fake_recon_index, total {index_shape[0]} items. {t:>4.2f} sec.')

    # Get test_ids
    print(f'test_id: \033[93m{test_ids}\033[0m,  ', end='')

    query_lookup = json.load(open(f'{emb_dir}/query_db_lookup.json', 'r'))
    ref_lookup = json.load(open(f'{emb_dir}/ref_db_lookup.json', 'r'))
    test_ids, max_test_seq_len = extract_test_ids(query_lookup)
    n_test = len(test_ids)
    with open('../data/gt_dict.json', 'r') as fp:
        gt = json.load(fp)

    print(f'n_test: \033[93m{n_test:n}\033[0m')

    """ Song-level search & evaluation """

    # Define metrics
    top1_exact = np.zeros((n_test, len(test_seq_len))).astype(np.int_)
    top3_exact = np.zeros((n_test, len(test_seq_len))).astype(np.int_)
    top10_exact = np.zeros((n_test, len(test_seq_len))).astype(np.int_)

    start_time = time.time()
    for ti, test_id in enumerate(test_ids):

        # Limit test_seq_len to max_test_seq_len
        max_len = int(max_test_seq_len[ti])
        max_query_len = test_seq_len[test_seq_len <= max_len]

        for si, sl in enumerate(max_query_len):

            hist = defaultdict(int)
            assert test_id <= len(query)
            q = query[test_id:(test_id + sl), :] # shape(q) = (length, dim)
            q_id = query_lookup[test_id].split("_")[0]    # query ID; split to remove the segment number

            # segment-level top k search for each segment
            S, I = index.search(q, k_probe)  

            # Flatten I and S while keeping their correspondence
            valid_indices = np.where(I >= 0) 
            candidates = np.unique(I[valid_indices])

            # Create a dictionary to map unique I values to their corresponding S values
            sims = {i: [] for i in candidates}
            for row, col in zip(*valid_indices):
                sims[I[row, col]].append(S[row, col])

            # Calculate the mean of the S values for each I value
            for i in sims:
                sims[i] = np.max(sims[i])

            """ Song-level match score """
            for ci, cid in enumerate(candidates):
                if cid < dummy_db_shape[0]:
                    continue
                match = ref_lookup[cid - dummy_db_shape[0]]
                # Ignore candidates which is the same as query file
                if match == q_id:
                    continue
                assert type(match) == str, f'{type(match)} is not str. See {ref_lookup}'
                candidate_seq = fake_recon_index[cid:(cid + sl), :] 
                if candidate_seq.shape[0] < sl:
                    q_match = q[:candidate_seq.shape[0], :] 
                else:
                    q_match = q     
                # score = np.mean(np.sum(q_match * candidate_seq, axis=1))
                score = sims[cid]
                hist[match] += score
                # hist[match] += 1

            """ Evaluate """
            pred = sorted(hist, key=hist.get, reverse=True)
            
            if pred:
                # Top-1 hit: 
                top1_exact[ti, si] = int(q_id in gt[pred[0]])
                # Top-3 hit:
                if any(q_id in gt[p] for p in pred[:3]):
                    top3_exact[ti, si] += 1
                # Top-10 hit:
                if any(q_id in gt[p] for p in pred[:10]):
                    top10_exact[ti, si] += 1


    # Summary 
    valid_mask = (test_seq_len <= max_test_seq_len[:, None])        # The mask preserves valid entries
    # print("Valid mask: ", valid_mask)
    top1_rate = 100. * np.nanmean(np.where(valid_mask, top1_exact, np.nan), axis=0)
    top3_rate = 100. * np.nanmean(np.where(valid_mask, top3_exact, np.nan), axis=0)
    top10_rate = 100. * np.nanmean(np.where(valid_mask, top10_exact, np.nan), axis=0)

    hit_rates = np.stack([top1_rate, top3_rate, top10_rate], axis=0)
    # del fake_recon_index, query, db
    del query, db

    # print(hit_rates)
    np.save(f'{emb_dir}/hit_rates.npy', hit_rates)

    np.save(f'{emb_dir}/raw_score.npy',
            np.concatenate(
                (top1_exact, top3_exact, top10_exact), axis=1))
    np.save(f'{emb_dir}/test_ids.npy', test_ids)
    print(f'Saved test_ids, hit-rates and raw score to {emb_dir}.')

    return hit_rates