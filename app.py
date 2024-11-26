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
    try:
        pagerank = nx.pagerank(graph, max_iter=100, tol=1e-08)
        return pagerank
    except Exception as e:
        st.error(f"Error computing PageRank: {e}")
        return None

def visualize_pagerank(graph, pagerank):
    """Visualize the PageRank of the graph."""
    pos = nx.spring_layout(graph, k=0.15, iterations=20)
    node_size = [pagerank[node] * 10000 for node in graph.nodes()]
    plt.figure(figsize=(12, 12))
    nx.draw(
        graph, pos, with_labels=True, node_size=node_size, 
        node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray'
    )
    plt.title('PageRank Visualization')
    st.pyplot(plt)

def load_available_datasets():
    """Load available datasets from the datasets folder."""
    datasets = [f for f in os.listdir(DATASETS_FOLDER) if f.endswith('.graphml')]
    return datasets

# Streamlit UI
st.title("PageRank Calculation and Visualization")
st.sidebar.header("Select or Upload a Graph")

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state['graph'] = None

# Provide options to either select a preloaded dataset or upload a file
option = st.sidebar.radio(
    "Choose an option",
    ('Select from available datasets', 'Upload your own file')
)

if option == 'Select from available datasets':
    datasets = load_available_datasets()
    if datasets:
        dataset = st.sidebar.selectbox("Select a dataset", datasets)
        if st.sidebar.button("Load Dataset"):
            filepath = os.path.join(DATASETS_FOLDER, dataset)
            try:
                graph = nx.read_graphml(filepath)
                st.session_state['graph'] = graph
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
            graph = nx.read_graphml(filepath)
            st.session_state['graph'] = graph
            st.success(f"Uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

# Retrieve the graph from session state
graph = st.session_state['graph']

if graph:
    # Display graph information
    st.write(f"Graph Information:\n{nx.info(graph)}")
    if not graph.is_directed():
        graph = graph.to_directed()
        st.warning("Converted undirected graph to directed for PageRank computation.")

    if st.button("Compute PageRank"):
        pagerank = compute_pagerank(graph)
        if pagerank:
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
