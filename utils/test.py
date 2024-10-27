
import utils
import numpy as np
import torch

def get_knn_indices(features,targets, topk=10):

    # from utils.evaluate_utils_fish import kmeans
    # kmeans(features, targets)

    import faiss
    # features = features.cpu().numpy()
    n, dim = features.shape[0], features.shape[1]
    print(n,dim)
    print('===============dim')
    index = faiss.IndexFlatIP(dim)  # index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)

    index.add(np.ascontiguousarray(features))
    distances, indices = index.search(np.ascontiguousarray(features), topk + 1)  # Sample itself is included

    # evaluate
    targets = targets.cpu().numpy()
    neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
    anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
    accuracy = np.mean(neighbor_targets == anchor_targets)

    return indices, accuracy

features = np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3]])

targets = np.array([0,1,0])

ind, acc = get_knn_indices(features,targets,1)
print(ind, acc)