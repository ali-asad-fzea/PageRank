import numpy as np

class Simplified_PageRank:
    def __init__(self, G, iteration=10000):
        self.G = G
        self.NUMBER_OF_ITERATION = iteration
        self.epsillon = 1e-8

    def run(self):
        n = len(self.G.nodes())
        matrix = np.zeros(shape=(n, n))
        for edge in self.G.edges():
            matrix[edge[0], edge[1]] = 1
        matrix = matrix / np.sum(matrix, axis=1).reshape(-1,1)
        vector = np.ones(n) / n

        NORM = []
        for _ in range(self.NUMBER_OF_ITERATION):
            new_vector = np.dot(matrix.T, vector)
            norm = np.linalg.norm(new_vector - vector, ord=1)
            NORM.append(norm)
            if norm <= self.epsillon : break
            vector = new_vector
        return new_vector, NORM