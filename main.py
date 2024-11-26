#!/usr/bin/env python
import networkx as nx
import time

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from Algorithms.Weighted import Weighted_PageRank
from Algorithms.Standard import Standard_PageRank
from Algorithms.Simplified import Simplified_PageRank
from Utils.Graph import Graph



def get_top_n_ranks(scores, int2node, n):
    # Create a list of (node, score) tuples and sort by score in descending order
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_n = [(node, score) for node, score in sorted_scores[:n]]
    return top_n

def standard_pageRank_with_multi_alpha(G_sp, int2node_sp):
    a1 = 0.01
    a2 = 0.3
    a3 = 0.7
    a4 = 0.85
    a1_v, a1_l2 = Standard_PageRank(G_sp, int2node_sp, a1, n_iter).run()
    a2_v, a2_l2 = Standard_PageRank(G_sp, int2node_sp, a2, n_iter).run()
    a3_v, a3_l2 = Standard_PageRank(G_sp, int2node_sp, a3, n_iter).run()
    a4_v, a4_l2 = Standard_PageRank(G_sp, int2node_sp, a4, n_iter).run()

    plt.figure(figsize=(10, 6))
    plt.plot(a1_l2, label='Damping Factor 0.01', color='blue')
    plt.plot(a2_l2, label='Damping Factor 0.3', color='green')
    plt.plot(a3_l2, label='Damping Factor 0.7', color='orange')
    plt.plot(a4_l2, label='Damping Factor 0.85', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('L2 Norm')
    plt.title('Convergence of PageRank with Different Damping Factors')
    plt.legend()
    # plt.grid(True)
    plt.show()

def multi_algo_convergence_graph(G_sp, int2node_sp, alpha=0.85, n_iter=10000, W=[[]], name_=""):
    top_k = 10


    std_rank, std_l2 = Standard_PageRank(G_sp, int2node_sp, alpha, n_iter).run()
    w_rank, w_l2 = Weighted_PageRank(G_sp, W, int2node_sp, alpha=alpha).run()
    sim_rank, sim_l2 = Simplified_PageRank(G_sp, iteration=n_iter).run()
    top_k_standard = get_top_n_ranks(std_rank, int2node_sp, top_k)
    top_k_weighted = get_top_n_ranks(w_rank, int2node_sp, top_k)
    top_k_simplified = get_top_n_ranks(sim_rank, int2node_sp, top_k)


    nx_pagerank_sim = nx.pagerank(G_sp, alpha=1, max_iter=n_iter)
    nx_pagerank_std = nx.pagerank(G_sp, alpha=alpha, max_iter=n_iter)
    nx_pagerank_wtd = nx.pagerank(nx.from_numpy_array(W, create_using=nx.DiGraph), alpha=alpha, max_iter=n_iter, weight='weight')
    nx_rank_sim = [(node, score) for node, score in nx_pagerank_sim.items()]
    nx_rank_std = [(node, score) for node, score in nx_pagerank_std.items()]
    nx_rank_wtd = [(node, score) for node, score in nx_pagerank_wtd.items()]
    top_k_nx_sim_rank = sorted(nx_rank_sim, key=lambda x: x[1], reverse=True)[:top_k]
    top_k_nx_std_rank = sorted(nx_rank_std, key=lambda x: x[1], reverse=True)[:top_k]
    top_k_nx_wtd_rank = sorted(nx_rank_wtd, key=lambda x: x[1], reverse=True)[:top_k]


    # Header for the table
    print(f"Dataset - Top {top_k} Ranks Comparison")
    print(f"{'Rank':<5} {'Simplified':<25}  {'NetworkX (Simple)':<25} {'Standard':<25} {'NetworkX (Standard)':<25} {'Weighted':<25} {'NetworkX (Weighted)':<25}")
    print('-' * 135)

    # Display top 5 for each algorithm
    for i in range(top_k):
        standard = f"{top_k_standard[i][0]}: {top_k_standard[i][1]:.8f}"
        weighted = f"{top_k_weighted[i][0]}: {top_k_weighted[i][1]:.8f}"
        simplified = f"{top_k_simplified[i][0]}: {top_k_simplified[i][1]:.8f}"
        nx_sim = f"{top_k_nx_sim_rank[i][0]}: {top_k_nx_sim_rank[i][1]:.8f}"
        nx_std = f"{top_k_nx_std_rank[i][0]}: {top_k_nx_std_rank[i][1]:.8f}"
        nx_wtd = f"{top_k_nx_wtd_rank[i][0]}: {top_k_nx_wtd_rank[i][1]:.8f}"
        print(f"{i + 1:<5} {simplified:<25} {nx_sim:<25} {standard:<25} {nx_std:<25} {weighted:<25} {nx_wtd:<25}")
    print('-' * 135)

    plt.figure()
    plt.plot(sim_l2, label='Simplified PageRank', color='blue')
    plt.plot(std_l2, label='Standard PageRank', color='green')
    plt.plot(w_l2, label='Weighted PageRank', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('L2 Norm')
    plt.title(f'Convergence on Dataset (alpha = {alpha})')
    plt.legend()
    plt.savefig(f'./docs/alpha_{alpha}_{ds}_conv.png', dpi = 400)


def algo_over_all_dataset(alpha=0.85, n_iter=1000):
    datasets = ["Reddit-2", "IMDB", "Wikipedia", "IISc"]
    files = ["./dataset/Reddit-graph-int-2.graphml", "./dataset/IMDB-graph-int-2.graphml", "./dataset/Wikipedia-graph-int-2.graphml", "./dataset/iisc-graph-int-2.graphml"]

    all_runtimes = []
    for name, filepath in zip(datasets[:], files[:]):
        G = nx.read_graphml(filepath)
        G_sp, int2node_sp, W = Graph.make_graph(G)
        runtimes = []

        start = time.time()
        std_rank, std_l2 = Standard_PageRank(G_sp, int2node_sp, alpha, n_iter).run()
        end = time.time()
        paper_time = end - start
        runtimes.append(paper_time)
        print(f"Standard : Done (Iteration {len(std_l2)})")

        start = time.time()
        w_rank, w_l2 = Weighted_PageRank(G_sp, W, alpha=alpha, int2node=int2node_sp).run()
        end = time.time()
        weighted_time = end - start
        runtimes.append(weighted_time)
        print(f"Weighted : Done (Iteration {len(w_l2)})")

        start = time.time()
        sim_rank, sim_l2 = Simplified_PageRank(G_sp, iteration=n_iter).run()
        end = time.time()
        simplified_time = end - start
        runtimes.append(simplified_time)
        print(f"Simplified : Done (Iteration {len(sim_l2)})")

        start = time.time()
        nx_pagerank_sim = nx.pagerank(G_sp, alpha=1, max_iter=n_iter)
        end = time.time()
        nx_time = end - start
        runtimes.append(nx_time)
        print("Networkx Simple : Done")

        start = time.time()
        nx_pagerank_std = nx.pagerank(G_sp, alpha=alpha, max_iter=n_iter)
        end = time.time()
        nx_time = end - start
        runtimes.append(nx_time)
        print("Network Standard : Done")

        start = time.time()
        nx_pagerank_wtd = nx.pagerank(nx.from_numpy_array(W, create_using=nx.DiGraph), alpha=alpha, max_iter=n_iter, weight='weight')
        end = time.time()
        nx_time = end - start
        runtimes.append(nx_time)
        print("Networkx Weighted : Done")

        # Store runtimes for the current dataset
        all_runtimes.append(runtimes)
        top_k = 10
        # Convert networkx pagerank result to format compatible with int2node_sp
        nx_rank_sim = [(node, score) for node, score in nx_pagerank_sim.items()]
        nx_rank_std = [(node, score) for node, score in nx_pagerank_std.items()]
        nx_rank_wtd = [(node, score) for node, score in nx_pagerank_wtd.items()]


        top_k_nx_sim_rank = sorted(nx_rank_sim, key=lambda x: x[1], reverse=True)[:top_k]
        top_k_nx_std_rank = sorted(nx_rank_std, key=lambda x: x[1], reverse=True)[:top_k]
        top_k_nx_wtd_rank = sorted(nx_rank_wtd, key=lambda x: x[1], reverse=True)[:top_k]

        top_k_standard = get_top_n_ranks(std_rank, int2node_sp, top_k)
        top_k_weighted = get_top_n_ranks(w_rank, int2node_sp, top_k)
        top_k_simplified = get_top_n_ranks(sim_rank, int2node_sp, top_k)

        # Header for the table
        print(f"Dataset - Top {top_k} Ranks Comparison")
        print(f"{'Rank':<5} {'Simplified':<25}  {'NetworkX (Simple)':<25} {'Standard':<25} {'NetworkX (Standard)':<25} {'Weighted':<25} {'NetworkX (Weighted)':<25}")
        print('-' * 135)

        # Display top 5 for each algorithm
        for i in range(top_k):
            standard = f"{top_k_standard[i][0]}: {top_k_standard[i][1]:.8f}"
            weighted = f"{top_k_weighted[i][0]}: {top_k_weighted[i][1]:.8f}"
            simplified = f"{top_k_simplified[i][0]}: {top_k_simplified[i][1]:.8f}"
            nx_sim = f"{top_k_nx_sim_rank[i][0]}: {top_k_nx_sim_rank[i][1]:.8f}"
            nx_std = f"{top_k_nx_std_rank[i][0]}: {top_k_nx_std_rank[i][1]:.8f}"
            nx_wtd = f"{top_k_nx_wtd_rank[i][0]}: {top_k_nx_wtd_rank[i][1]:.8f}"
            print(f"{i + 1:<5} {simplified:<25} {nx_sim:<25} {standard:<25} {nx_std:<25} {weighted:<25} {nx_wtd:<25}")
        print('-' * 135)
        
    # Plotting the grouped bar chart
    labels = ['Standard\nPageRank', 'Weighted\nPageRank', 'Simplified\nPageRank', "networkx\n(Simple)", "networkx\n(Standard)", "networkx \n(Weighted)"]
    colors = ['blue', 'green', 'red', "yellow", "black", "orange"]
    x = range(len(labels))
    width = 0.2  # Width of each bar

    plt.figure(figsize=(12, 8))
    for i, (dataset, runtimes) in enumerate(zip(datasets, all_runtimes)):
        plt.bar([p + i * width for p in x], runtimes, width=width, label=dataset, color=colors[i % len(colors)])

    plt.xlabel('Algorithm')
    plt.ylabel('Run Time (seconds)')
    plt.title('Run Time Comparison of PageRank Algorithms across Datasets')
    plt.xticks([p + 1.5 * width for p in x], labels)
    plt.legend(datasets)
    plt.grid(axis='y')
    plt.savefig(f'./docs/Figure_bar.png', dpi = 1000)


if __name__ == "__main__":
    alpha = 0.85
    n_iter = 10000
    # G = nx.read_graphml("./dataset/Reddit-graph-str-2.graphml")
    # G = Graph.generate_sparse_graph(add_edge=True)
    # G_sp, int2node_sp, W = Graph.make_graph(G)

    datasets = ["Reddit-2", "IMDB-2", "Wikipedia-2", "IISc-2"]
    files = ["./dataset/Reddit-graph-int-2.graphml", "./dataset/IMDB-graph-int-2.graphml", "./dataset/Wikipedia-graph-int-2.graphml", "./dataset/iisc-graph-int-2.graphml"]
    for ds, f in zip(datasets, files):
        G = nx.read_graphml(f)
        G_sp, int2node_sp, W = Graph.make_graph(G)
        for al in [0.25, 0.50, 0.75, 0.85]:
            multi_algo_convergence_graph(G_sp, int2node_sp, al, n_iter, W, ds)
    algo_over_all_dataset(alpha=alpha, n_iter=n_iter)



