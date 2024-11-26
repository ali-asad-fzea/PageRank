import numpy as np

class Standard_PageRank:
    
    def __init__(self, G, int2node, alpha=0.85, n_iter=10000):
        self.G = G
        self.int2node = int2node
        self.alpa = alpha
        self.n_iter = n_iter
        self.epsillon = 1e-8

    def make_pagerank_matrix(self, G, alpha):
        n_nodes = len(G.nodes())
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for edge in G.edges():
            adj_matrix[edge[0], edge[1]] = 1

        row_sums = np.sum(adj_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero for dangling nodes
        tran_matrix = adj_matrix / row_sums
        
        # Random surfing matrix (uniform probability distribution)
        random_surf = np.ones((n_nodes, n_nodes)) / n_nodes
        
        # Handle dangling nodes
        absorbing_nodes = np.zeros(n_nodes)
        for node in G.nodes():
            if len(G.out_edges(node)) == 0:
                absorbing_nodes[node] = 1
        absorbing_node_matrix = np.outer(absorbing_nodes, np.ones(n_nodes)) / n_nodes
        stochastic_matrix = tran_matrix + absorbing_node_matrix
        pagerank_matrix = alpha * stochastic_matrix + (1 - alpha) * random_surf
        return pagerank_matrix
    

    def random_walk(self, G, alpha, n_iter):
        n_nodes = len(G.nodes())

        new_state = np.ones(n_nodes) / n_nodes
        pagerank_matrix = self.make_pagerank_matrix(G, alpha)
        
        L2 = []
        for i in range(n_iter):
            old_state = new_state
            new_state = pagerank_matrix.T @ old_state
            l2 = np.linalg.norm(new_state - old_state, ord=1)
            L2.append(l2)
            if l2 < self.epsillon: break
        return new_state, L2
    
    def run(self):
        final_probs, L2 = self.random_walk(self.G, self.alpa, self.n_iter)
        assert len(final_probs) == len(self.G.nodes())
        # assert np.allclose(np.sum(final_probs), 1)
        return final_probs, L2
