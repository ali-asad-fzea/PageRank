import numpy as np

class Simulation_PageRank:
    def __init__(self, G, int2node=None, iteration=10000):
        self.graph = G
        self.iteration = 0
        self.initial_points = None
        self.int2node = int2node
        self.NUMBER_OF_ITERATION = iteration
        self.epsillon = 1e-8
        self.initialization()

    def initialization(self):
        num_nodes = self.graph.number_of_nodes()
        self.initial_points = np.full(num_nodes, 1 / num_nodes)

    def update(self, G, prev_points):
        self.iteration += 1
        num_nodes = G.number_of_nodes()
        update_point = np.zeros(num_nodes)

        for node in G.nodes():
            out_edges = list(G.out_edges(node))
            if out_edges:
                share = prev_points[node] / len(out_edges)
                for _, target in out_edges:
                    update_point[target] += share
            else:
                update_point[node] += prev_points[node]
        return update_point

    def run(self):
        prev_points = self.initial_points
        iteration = 0
        L2 = []
        while iteration <= self.NUMBER_OF_ITERATION:
            update_points = self.update(self.graph, prev_points)
            l2 = np.linalg.norm(update_points-prev_points)
            L2.append(l2)
            if np.all(np.abs(update_points - prev_points) <= self.epsillon): break
            prev_points, iteration = update_points, iteration+1
        return prev_points, L2