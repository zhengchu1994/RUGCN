from tqdm import tqdm
import numpy as np
import numpy.random as npr
from collections import defaultdict

class Walker:
    def __init__(self, adj, features, idx_train, idx_val, idx_test, num_walks, walk_length, p=1.0, q=1.0):
        self.adj = adj.tocsr()
        self.features = features
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.process_probs().simulate_walk(self.num_walks, walk_length)

    def simulate_walk(self, num_walks, walk_length):
        walks = defaultdict(list)
        for i in tqdm(range(num_walks)):
            print(f"Walk iteration: {i + 1}")
            index = np.concatenate([self.idx_val, self.idx_test])
            for idx in index:
                res = self.node2vec_walk(walk_length, idx)
                walks[idx] += res[1:]
        self.walks = walks

    def node2vec_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            _, neighs = self.adj[cur, :].nonzero()
            if len(neighs) > 0:
                if len(walk) == 1:
                    res = neighs[
                        self.alias_draw(self.alias_nodes[cur][0]
                                        , self.alias_nodes[cur][1])]
                    if res in self.idx_train:
                        break
                    else:
                        walk.append(res)
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    res = neighs[
                        self.alias_draw(self.alias_edges[edge][0],
                                        self.alias_edges[edge][1])]
                    if res in self.idx_train:
                        break
                    else:
                        walk.append(res)
            else:
                break
        return walk

    def process_probs(self):
        adj = self.adj
        alias_nodes = {}
        size = adj.shape[0]
        for cur_row in range(size):
            _, cur_col = adj[cur_row, :].nonzero()
            unnormalized_probs = np.ones(len(cur_col))
            deno = unnormalized_probs.sum()
            normalized_probs = unnormalized_probs / deno
            alias_nodes[cur_row] = self.alias_setup(normalized_probs)

        alias_edges = {}
        raw_g, col_g = adj.nonzero()
        for edge in zip(raw_g, col_g):
            alias_edges[edge] = self.get_alias_edges(edge[0], edge[1])
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return self

    def get_alias_edges(self, src, dst):
        adj = self.adj
        unnormalized_probs = []
        _, neighs = self.adj[dst, :].nonzero()
        for neigh in neighs:
            if neigh == src:
                unnormalized_probs.append(1.0 / self.p)
            elif adj[(src, neigh)] or adj[(neigh, src)]:
                unnormalized_probs.append(1.0)
            else:
                unnormalized_probs.append(1.0 / self.q)
            deno = sum(unnormalized_probs)
            normalized_probs = [float(prob) / deno for prob in unnormalized_probs]
        return self.alias_setup(normalized_probs)

    def alias_setup(self, probs):
        '''
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        q = np.zeros(K, dtype=np.float32)
        J = np.zeros(K, dtype=np.int32)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def alias_draw(self, J, q):
        K = len(J)

        # Draw from the overall uniform mixture.
        kk = int(np.floor(npr.rand() * K))

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        if npr.rand() < q[kk]:
            return kk
        else:
            return J[kk]
