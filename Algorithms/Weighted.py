import numpy as np


class Weighted_PageRank:
    def __init__(self, graph, W, int2node, alpha=0.85, max_iter=1000):
        self.alpha = alpha
        self.epsilon = 1e-8
        self.graph = graph
        self.nodes = sorted(graph.nodes())
        self.number_nodes = len(self.nodes)
        self.number_edges = len(self.graph.edges)
        self.max_iter = max_iter
        self.W = W
        self.int2node = int2node

    def get_weighted_adj_matrix(self):
        denominator = np.sum(self.W,axis=1)
        denominator = np.where(denominator == 0, 1., denominator)
        return (self.W.T/denominator)

    def run(self):
        state = np.ones((self.number_nodes, 1)) / self.number_nodes
        Id = np.ones((self.number_nodes, 1))

        norms = []
        for it in range(self.max_iter):
            new_state = self.alpha * np.matmul(self.get_weighted_adj_matrix(), state) + (1 - self.alpha) * Id / self.number_nodes
            new_state /= np.sum(new_state)
            norm = np.linalg.norm(new_state - state, ord=1)
            norms.append(norm)
            if norm <= self.epsilon: break
            state = new_state
        return  [float(rank) for rank in new_state.flatten()], norms
