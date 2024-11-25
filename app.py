import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle

# Configuration
DATASETS_FOLDER = './datasets'
UPLOAD_FOLDER = './uploads'

# Ensure necessary folders exist
os.makedirs(DATASETS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compute_pagerank(graph):
    """Compute PageRank for a given graph."""
    pagerank = nx.pagerank(graph, max_iter=100, tol=1e-08)
    return pagerank

def visualize_pagerank(graph, pagerank):
    """Visualize the PageRank of the graph."""
    pos = nx.spring_layout(graph, k=0.15, iterations=20)
    node_size = [pagerank[node] * 10000 for node in graph.nodes()]
    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, with_labels=True, node_size=node_size, node_color='skyblue', 
            font_size=10, font_weight='bold', edge_color='gray')
    plt.title('PageRank Visualization')
    st.pyplot()

def load_available_datasets():
    """Load available datasets from the datasets folder."""
    datasets = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.graphml')]
    return datasets

# Streamlit UI
st.title("PageRank Calculation and Visualization")
st.sidebar.header("Select or Upload a Graph")

# Provide options to either select a preloaded dataset or upload a file
option = st.sidebar.radio(
    "Choose an option",
    ('Select from available datasets', 'Upload your own file')
)

graph = None
pagerank = None

if option == 'Select from available datasets':
    datasets = load_available_datasets()
    if datasets:
        dataset = st.sidebar.selectbox("Select a dataset", datasets)
        if st.sidebar.button("Load Dataset"):
            filepath = os.path.join(DATASETS_FOLDER, dataset)
            graph = nx.read_graphml(filepath)
            st.success(f"Loaded dataset: {dataset}")
    else:
        st.warning("No datasets available in the datasets folder.")

elif option == 'Upload your own file':
    uploaded_file = st.sidebar.file_uploader("Upload a GraphML file", type=["graphml"])
    if uploaded_file:
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        graph = nx.read_graphml(filepath)
        st.success(f"Uploaded file: {uploaded_file.name}")

# If a graph is loaded, compute and visualize PageRank

if graph is not None:
    if st.button("Compute PageRank"):
        st.write("Computing PageRank of",graph)
        pagerank = compute_pagerank(graph)
        st.write("PageRank Values:")
        st.write(pagerank)
        visualize_pagerank(graph, pagerank)

        # Save PageRank values for future use
        with open('ranks.pkl', 'wb') as f:
            pickle.dump(pagerank, f)

        # Show sorted PageRank results
        sorted_pagerank = dict(sorted(pagerank.items(), key=lambda item: item[1], reverse=True))
        st.write("Sorted PageRank:")
        st.write(sorted_pagerank)
else:
    st.info("Please select a dataset or upload a file to proceed.")
