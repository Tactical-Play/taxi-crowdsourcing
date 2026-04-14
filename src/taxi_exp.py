from collections import defaultdict
import math

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371.0  # Radius of Earth in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# -------------------------------
# BUILD LINK GRAPH
# -------------------------------
def build_link_graph(trip_dict):
    link_graph = defaultdict(lambda: defaultdict(int))

    for _, (o, d) in trip_dict.items():
        link_graph[o][d] += 1

    return link_graph


# -------------------------------
# GRAPH ENTROPY (Eq. 1)
# -------------------------------
def compute_graph_entropy(link_graph):
    freq = {}
    total_flow = 0

    for v in link_graph:
        f = sum(link_graph[v].values())
        freq[v] = f
        total_flow += f

    H = 0
    for v in freq:
        if freq[v] == 0:
            continue
        p = freq[v] / total_flow
        H -= p * math.log(p)

    return H, freq, total_flow


# -------------------------------
# NODE ENTROPY (Eq. 2)
# -------------------------------
def compute_node_entropy(link_graph):
    entropy = {}

    for v in link_graph:
        total = sum(link_graph[v].values())
        if total == 0:
            entropy[v] = 0
            continue

        H = 0
        for u in link_graph[v]:
            p = link_graph[v][u] / total
            H -= p * math.log(p)

        entropy[v] = H

    return entropy


# -------------------------------
# REMOVE NODE FROM GRAPH
# -------------------------------
def remove_node(link_graph, v_remove):
    new_graph = {}

    for v in link_graph:
        if v == v_remove:
            continue

        new_graph[v] = {}
        for u in link_graph[v]:
            if u != v_remove:
                new_graph[v][u] = link_graph[v][u]

    return new_graph


# -------------------------------
# RANK NODES (Algorithm 1)
# -------------------------------
def rank_nodes(link_graph, top_k=10000):

    # Step 1: frequency
    freq = {v: sum(link_graph[v].values()) for v in link_graph}

    # Step 2: select top-k
    top_nodes = sorted(freq, key=freq.get, reverse=True)[:top_k]

    # Step 3: subgraph
    subgraph = {v: link_graph[v] for v in top_nodes}

    # Step 4: node entropy
    node_entropy = compute_node_entropy(subgraph)

    influence = {}

    # Step 5: compute IF for each node
    for i,v in enumerate(top_nodes):
        if i % 100 == 0:
            print(f"Processing node {i}/{len(top_nodes)}")
        subgraph_removed = remove_node(subgraph, v)

        EnG, _, _ = compute_graph_entropy(subgraph_removed)

        if EnG == 0:
            influence[v] = 0
        else:
            influence[v] = node_entropy[v] / EnG

    # Step 6: rank
    ranked_nodes = sorted(influence, key=influence.get, reverse=True)

    return ranked_nodes


# -------------------------------
# HUB IDENTIFICATION (Algorithm 2)
# -------------------------------
def identify_hubs(G, ranked_nodes, threshold_km=3):

    hubs = []
    remaining = list(ranked_nodes)

    while remaining:
        if len(hubs) % 100 == 0:
            print(f"Hubs selected: {len(hubs)}")
        v = remaining.pop(0)
        hubs.append(v)

        lon1, lat1 = G.nodes[v]['pos']

        new_remaining = []

        for u in remaining:

            lon2, lat2 = G.nodes[u]['pos']

            dist = haversine_distance(lat1, lon1, lat2, lon2)

            if dist >= threshold_km:
                new_remaining.append(u)

        remaining = new_remaining

    return hubs