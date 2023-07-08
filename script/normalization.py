import faiss
import numpy as np
from sklearn.cluster import KMeans

# np.set_printoptions(precision=3, suppress=True)

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


def ip_gnd(base, query, k):
    base_dim = base.shape[1]
    index = faiss.IndexFlatIP(base_dim)
    index.add(base)
    gnd_distance, gnd_idx = index.search(query, k)
    return gnd_idx, gnd_distance


if __name__ == '__main__':
    user_l = np.array(
        [[2.7, 2.1, 8.6], [7.5, 5.0, 3.4], [8.2, 2.5, 3.1], [4.4, 3.9, 4.6], [4.0, 0.6, 2.8], [8.3, 9.4, 8.2]],
        dtype=np.float32)
    norm_l = [np.linalg.norm(vecs) for vecs in user_l]
    user_l = np.array([user_l[i] / norm_l[i] for i in range(len(user_l))])
    print(user_l)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(user_l)
    print(kmeans.labels_)

    item_l = np.array([[3.6, 3.5, 3.9], [5.0, 2.8, 4.2], [9.1, 7.4, 1.8], [9.0, 5.9, 1.3], [9.2, 4.3, 3.5],
                       [3.0, 9.8, 5.7]], dtype=np.float32)
    idx, dist = ip_gnd(item_l, user_l, 6)
    print(idx + 1)
    print(dist)
