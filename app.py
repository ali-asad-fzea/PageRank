import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import os
import pickle
import streamlit as st

# Create a directory for uploads
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def merge_url_schemes2(input_dict):
    result = {}
    for url, value in input_dict.items():
        base_url = url.split('://')[-1]
        if base_url in result:
            result[base_url] += value
        else:
            result[base_url] = value
    final_result = {}
    for url in result:
        final_result[f"https://{url}"] = result[url]
    return final_result

def merge_url_schemes_with_subdomains(input_dict):
    result = {}
    for url, value in input_dict.items():
        parsed_url = urlparse(url)
        domain_parts = parsed_url.netloc.split('.')
        if len(domain_parts) > 2:
            root_domain = '.'.join(domain_parts[-2:])
        else:
            root_domain = parsed_url.netloc
        normalized_url = f"https://{root_domain}"
        if normalized_url in result:
            result[normalized_url] += value
        else:
            result[normalized_url] = value
    return result

def merge_url_schemes(input_dict):
    result = {}
    for url, value in input_dict.items():
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if base_url in result:
            result[base_url] += value
        else:
            result[base_url] = value
    return merge_url_schemes2(result)

def crawl(start_url, max_depth=3):
    visited = set()
    graph = nx.DiGraph()

    def crawl_page(url, depth):
        if depth > max_depth or url in visited:
            return
        visited.add(url)
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [a['href'] for a in soup.find_all('a', href=True)]
            for link in links:
                if link.startswith('http') or link.startswith('https'):
                    graph.add_edge(url, link)
                    crawl_page(link, depth + 1)
        except requests.exceptions.RequestException:
            pass

    crawl_page(start_url, 0)
    return graph

def compute_pagerank(graph):
    pagerank = nx.pagerank(graph, max_iter=100, tol=1e-08)
    return pagerank

def visualize_pagerank(graph, pagerank):
    pos = nx.spring_layout(graph, k=0.15, iterations=20)
    node_size = [pagerank[node] * 10000 for node in graph.nodes()]
    plt.figure(figsize=(12, 12))
    nx.draw(graph, pos, with_labels=True, node_size=node_size, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title('PageRank Visualization')
    st.pyplot()

def normalize_base_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def merge_nodes(graph):
    normalize_mapping = {}
    for node in list(graph.nodes):
        normalized_id = normalize_base_url(node)
        if normalized_id in normalize_mapping:
            existing_node = normalize_mapping[normalized_id]
            graph = nx.contracted_nodes(graph, existing_node, node, self_loops=False)
        else:
            normalize_mapping[normalized_id] = node
    return graph

# Streamlit UI
st.title("PageRank Calculation and Visualization")
st.sidebar.header("Input")
url = st.sidebar.text_input("Enter a URL to crawl", "")

file = st.sidebar.file_uploader("Upload a GraphML File", type=["graphml"])

if url:
    graph = crawl(url)
    pagerank = compute_pagerank(graph)
    st.write("PageRank Values:")
    st.write(pagerank)
    visualize_pagerank(graph, pagerank)

elif file:
    if file.name.endswith('.graphml'):
        filepath = os.path.join(UPLOAD_FOLDER, file.name)
        with open(filepath, "wb") as f:
            f.write(file.getbuffer())
        graph = nx.read_graphml(filepath)
        pagerank = compute_pagerank(graph)
        st.write("PageRank Values:")
        st.write(pagerank)
        visualize_pagerank(graph, pagerank)
        
        # Save the pagerank values for future use
        with open('ranks.pkl', 'wb') as f:
            pickle.dump(pagerank, f)

        # Merge URLs in the pagerank
        pagerank = merge_url_schemes(pagerank)
        pagerank = dict(reversed(sorted(pagerank.items(), key=lambda item: item[1])))
        st.write("Merged PageRank:")
        st.write(pagerank)
else:
    st.write("Please input a URL or upload a GraphML file to begin.")

