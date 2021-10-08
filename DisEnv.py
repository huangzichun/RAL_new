import time
import numpy as np
from sklearn.neighbors import KDTree
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class DisEnv:
    def __init__(self, X, leaf_size=40, k=3, label_dict=None):
        self.X = X
        self.leaf_size = leaf_size
        # since KDTree return top k neighbours, including the node itself.
        self.k = k + 1
        self.tree = self._init_kdtree()
        # neighbour similarity and ids
        self.dists, self.ids = None, None

        # if his neighbour has label, {id:label}
        self.label_dict = label_dict
        self.label_ind = np.zeros((len(self.X), k))

        # if he shares the same label with his neighbour (not considering non-label)
        self.label_share = np.zeros((len(self.X), k))

    def _init_kdtree(self):
        return KDTree(self.X, leaf_size=self.leaf_size)

    def _get_k_node(self, x):
        dist, ind = self.tree.query(x, k=self.k)
        return dist, ind

    def _init_neighbour_matrix(self):
        dists, ids = [], []
        for x in self.X:
            dist, id = self._get_k_node(x[np.newaxis,:])
            dist = [1.0 - sigmoid(dis) for dis in np.squeeze(dist, 0)]
            dists.append(dist)
            ids.append(np.squeeze(id, 0))
        return np.array(dists)[:, 1:], np.array(ids)[:, 1:]

    def _init_neighbour_label(self):
        self._update_neighbour_label(self.label_dict)

    def _init_shared_neighbour_label(self):
        self._update_shared_neighbour_label(self.label_dict)

    def init_env(self):
        self.dists, self.ids = self._init_neighbour_matrix()

        # if his neighbour has label
        self._init_neighbour_label()

        # if he shares the same label with his neighbour (not considering non-label)
        self._init_shared_neighbour_label()
        print("DisEnv initialized..")

    def _update_neighbour_label(self, label_dict_new):
        if label_dict_new and len(label_dict_new) > 0:
            for lid in list(label_dict_new.keys()):
                self.label_ind[self.ids == lid] = 1.0

    def _update_shared_neighbour_label(self, label_dict_new):
        if not label_dict_new or len(label_dict_new) <= 0:
            return
        for lid in label_dict_new.keys():
            my_label = label_dict_new.get(lid)
            my_neighbors = self.ids[lid]
            for my_neighbor_idx in range(len(my_neighbors)):
                my_neighbor = my_neighbors[my_neighbor_idx]
                if my_neighbor in label_dict_new:
                    my_neighbor_label = label_dict_new.get(my_neighbor)
                    self.label_share[lid][my_neighbor_idx] = 1 if my_neighbor_label == my_label else -1

    def update_env(self, label_dict_new):
        assert len(label_dict_new) > 0, "labeled information is required"
        # if his neighbour has label
        self._update_neighbour_label(label_dict_new)

        # if he shares the same label with his neighbour (not considering non-label)
        self._update_shared_neighbour_label(label_dict_new)

    def _get_max_by_col(self, x):
        return x.max(axis=1)

    def _get_gap_by_col(self, x):
        same = x.max(axis=1)
        same = np.where(same, same, np.nan)

        diff = -1.0 * x.min(axis=1)
        diff = np.where(diff, diff, np.nan)

        res = np.abs(same - diff)
        res[np.isnan(res)] = -1.0
        return res

    def get_stats(self):
        # fixed
        # nearest labeled neighbour, bigger value
        nearest_labeled_neighbour = self._get_max_by_col(self.dists * self.label_ind)

        # nearest unlabeled neighbour
        nearest_unlabeled_neighbour = self._get_max_by_col(self.dists * (1 - self.label_ind))

        # nearest positive_label - negative_label
        nearest_label_type_neighbour = self._get_gap_by_col(self.dists * self.label_share)

        return nearest_labeled_neighbour, nearest_unlabeled_neighbour, nearest_label_type_neighbour


if __name__ == "__main__":
    label_dict = {}
    X = np.random.rand(5300, 50)
    # start_time = time.time()
    dis_env = DisEnv(X, leaf_size=400, k=3, label_dict=label_dict)
    dis_env.init_env()
    print(dis_env.get_stats())