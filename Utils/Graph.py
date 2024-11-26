import numpy as np
import networkx as nx
import random


class Graph:
    @staticmethod
    def make_graph(G, tras_zero_degree=True):        
        # Convert to directed graph if not already
        print("Number of Nodes : ", len(G.nodes()))
        if not nx.is_directed(G):
            print('Graph converted to directed..')
            G = G.to_directed() 

        # Relabel nodes as integers
        n_unique_nodes = len(set(G.nodes()))
        node2int = dict(zip(set(G.nodes()), range(n_unique_nodes)))
        int2node = {v: k for k, v in node2int.items()}

        # Remove isolated nodes (nodes with no edges)
        G = nx.relabel_nodes(G, node2int)
        isolated_nodes = list(nx.isolates(G))
        print("Isolated Nodes:", len(isolated_nodes))
        G.remove_nodes_from(isolated_nodes)

        if not tras_zero_degree:
            return G, int2node, Graph.build_weight_matrix(G)
        
        # Find nodes with zero out-degree (dangling nodes)
        dangling_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
        print("Dangling Nodes (Zero out-degree):", len(dangling_nodes))

        # Add a random outgoing edge for each dangling node
        all_nodes = list(G.nodes())
        for node in dangling_nodes:
            for _ in range(10):
                target_node = random.choice(all_nodes)
                while target_node == node:
                    target_node = random.choice(all_nodes)
                G.add_edge(node, target_node)

        W = Graph.build_weight_matrix(G)
        return G, int2node, W
    
    @staticmethod
    def generate_sparse_graph(min_children=2, max_children=5, max_depth=6, add_edge = False, edge_fraction=0.1, shuffel=1336):
        random.seed(shuffel)
        G = nx.DiGraph()
        G.add_node(0)
        current_level = [0]
        node_count = 1

        for depth in range(max_depth):
            next_level = []
            for parent in current_level:
                num_children = random.randint(min_children, max_children)
                for _ in range(num_children):
                    G.add_node(node_count)
                    G.add_edge(parent, node_count)
                    next_level.append(node_count)
                    node_count += 1
            current_level = next_level

        # Add last level node to any random node in graph
        for c in current_level:
            v = c
            while c == v:
                v = random.randint(0, node_count - 1)
            G.add_edge(c, v)

        # Add additional random edges to make the graph sparse
        if(add_edge):
            num_nodes = len(G.nodes())
            max_edges = int(edge_fraction * num_nodes * (num_nodes - 1) / 2)
            added_edges = 0
            while added_edges < max_edges:
                u = random.randint(0, num_nodes - 1)
                v = random.randint(0, num_nodes - 1)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
                    added_edges += 1
        return G

    @staticmethod
    def build_weight_matrix(G):
        indegrees = dict(G.in_degree())
        outdegrees = dict(G.out_degree())
        W = np.zeros((len(G.nodes), len(G.nodes)), dtype=np.float16)
        edges = np.array(G.edges)
        nodes = sorted(G.nodes())
        for node in nodes:
            out_nodes = np.unique(edges[edges[:,0] == node][:,1])
            I = np.array([indegrees.get(i, 0) for i in out_nodes])
            O = np.array([outdegrees.get(i, 0) for i in out_nodes])
            w_in, w_out = I/np.sum(I), O/np.sum(O)
            col_idx = [nodes.index(node) for node in out_nodes]
            W[nodes.index(node),col_idx] = w_in*w_out
        return W