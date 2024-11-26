import streamlit as st
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt

# Import custom PageRank implementations
from Algorithms.Weighted import Weighted_PageRank
from Algorithms.Standard import Standard_PageRank
from Algorithms.Simplified import Simplified_PageRank
from Utils.Graph import Graph

# Configuration
DATASETS_FOLDER = './datasets'
UPLOAD_FOLDER = './uploads'

# Ensure necessary folders exist
os.makedirs(DATASETS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def visualize_pagerank(graph, pagerank, title="PageRank Visualization"):
    """Visualize the PageRank of the graph."""
    pos = nx.spring_layout(graph, k=0.15, iterations=20)
    node_size = [pagerank[node] * 10000 for node in graph.nodes()]
    plt.figure(figsize=(12, 12))
    nx.draw(
        graph, pos, with_labels=True, node_size=node_size,
        node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray'
    )
    plt.title(title)
    st.pyplot(plt)

def get_top_n_ranks(scores, int2node, n):
    # Create a list of (node, score) tuples and sort by score in descending order
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_n = [(node, score) for node, score in sorted_scores[:n]]
    return top_n

# Streamlit UI
st.title("Custom PageRank Algorithms with Visualization")
st.sidebar.header("Options")

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state['graph'] = None
    st.session_state['int2node_sp'] = None
    st.session_state['G_sp'] = None
    st.session_state['W'] = None

# Load or upload dataset
option = st.sidebar.radio(
    "Choose an option",
    ('Select from available datasets', 'Upload your own file')
)

if option == 'Select from available datasets':
    datasets = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.graphml')]
    if datasets:
        dataset = st.sidebar.selectbox("Select a dataset", datasets)
        if st.sidebar.button("Load Dataset"):
            filepath = os.path.join(DATASETS_FOLDER, dataset)
            try:
                G = nx.read_graphml(filepath)
                G_sp, int2node_sp, W = Graph.make_graph(G)
                st.session_state['graph'] = G
                st.session_state['G_sp'] = G_sp
                st.session_state['int2node_sp'] = int2node_sp
                st.session_state['W'] = W
                st.success(f"Loaded dataset: {dataset}")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
    else:
        st.warning("No datasets available in the datasets folder.")

elif option == 'Upload your own file':
    uploaded_file = st.sidebar.file_uploader("Upload a GraphML file", type=["graphml"])
    if uploaded_file:
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            G = nx.read_graphml(filepath)
            G_sp, int2node_sp, W = Graph.make_graph(G)
            st.session_state['graph'] = G
            st.session_state['G_sp'] = G_sp
            st.session_state['int2node_sp'] = int2node_sp
            st.session_state['W'] = W
            st.success(f"Uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

# Retrieve graph and transformed data from session state
graph = st.session_state['graph']
G_sp = st.session_state['G_sp']
int2node_sp = st.session_state['int2node_sp']
W = st.session_state['W']

if graph:
    st.write(f"Graph Information:\n{graph}")

    # Algorithm and parameters selection
    algo_option = st.sidebar.selectbox("Select a PageRank Algorithm", ['Standard', 'Weighted', 'Simplified'])
    alpha = st.sidebar.slider("Set Alpha (Damping Factor)", 0.1, 1.0, 0.85, 0.01)
    n_iter = st.sidebar.number_input("Set Number of Iterations", min_value=10, max_value=1000, value=100, step=10)
    top_k = st.sidebar.number_input("Set Top K Nodes", min_value=1, max_value=100, value=10, step=1)

    if st.button("Compute PageRank"):
        if algo_option == 'Standard':
            std_rank, _ = Standard_PageRank(G_sp, int2node_sp, alpha, n_iter).run()
            top_k_result = get_top_n_ranks(std_rank, int2node_sp, top_k)
            st.write(f"Top {top_k} nodes for Standard PageRank:")
            st.write(top_k_result)
            visualize_pagerank(graph, {node: std_rank[int(node)] for node in graph.nodes()}, "Standard PageRank")

        elif algo_option == 'Weighted':
            w_rank, _ = Weighted_PageRank(G_sp, W, int2node_sp, alpha).run()
            top_k_result = get_top_n_ranks(w_rank, int2node_sp, top_k)
            st.write(f"Top {top_k} nodes for Weighted PageRank:")
            st.write(top_k_result)
            visualize_pagerank(graph, {node: w_rank[int(node)] for node in graph.nodes()}, "Weighted PageRank")

        elif algo_option == 'Simplified':
            sim_rank, _ = Simplified_PageRank(G_sp, iteration=n_iter).run()
            top_k_result = get_top_n_ranks(sim_rank, int2node_sp, top_k)
            st.write(f"Top {top_k} nodes for Simplified PageRank:")
            st.write(top_k_result)
            visualize_pagerank(graph, {node: sim_rank[int(node)] for node in graph.nodes()}, "Simplified PageRank")
else:
    st.info("Please select a dataset or upload a file to proceed.")
