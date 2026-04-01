from collections import Counter
import heapq

def dijkstra_forward(G, source, threshold):
    distances = {}
    visited = set()

    # Min-heap: (distance, node)
    heap = [(0, source)]

    while heap:
        dist_u, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        if dist_u > threshold:
            break  # stop further exploration

        distances[u] = dist_u

        for v in G.neighbors(u):
            if v not in visited:
                edge_weight = G[u][v].get('weight')
                new_dist = dist_u + edge_weight
                if new_dist <= threshold:
                    heapq.heappush(heap, (new_dist, v))

    return distances

def dijkstra_reverse(G, source, threshold):
    distances = {}
    visited = set()

    # Min-heap: (distance, node)
    heap = [(0, source)]

    while heap:
        dist_u, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        if dist_u > threshold:
            break  # stop further exploration

        distances[u] = dist_u

        for v in G.predecessors(u):
            if v not in visited:
                edge_weight = G[v][u].get('weight')  # default to 1 as no weight
                new_dist = dist_u + edge_weight
                if new_dist <= threshold:
                    heapq.heappush(heap, (new_dist, v))

    return distances

def get_top_nodes_by_traffic(trip_dict, top_n=1000,threshold=3):
    node_counter = Counter()
    dijkstra_cache_start={}
    dijkstra_cache_end={}
    for start, end in trip_dict.values():
        if start not in dijkstra_cache_start: dijkstra_cache_start[start]=dijkstra_reverse(G, start, threshold)
        for key in dijkstra_cache_start[start].keys():
            node_counter[key]+=1

        if end not in dijkstra_cache_end: dijkstra_cache_end[end]=dijkstra_forward(G, end, threshold)
        for key in dijkstra_cache_end[end].keys():
            node_counter[key]+=1
    # Get top `top_n` nodes by traffic
    top_nodes = [node for node, _ in node_counter.most_common(top_n)]
    return top_nodes

def compute_forward_shortest_paths(G, S, threshold):
    results = {}
    i_count=0
    for source in S:
        i_count+=1
        if i_count%100==0: print(f"Running Dijkstra for hub no. {i_count} of {len(S)}")
        distances = dijkstra_forward(G, source, threshold)
        results[source] = distances
    return results

def compute_backward_shortest_paths(G, S, threshold):
    results = {}
    i_count=0
    for source in S:
        i_count+=1
        if i_count%100==0: print(f"Running Dijkstra for hub no. {i_count} of {len(S)}")
        distances = dijkstra_reverse(G, source, threshold)
        results[source] = distances
    return results

def build_OC_EC_optimized(trip_dict, forward_distances, backward_distances):
    # -------------------------------------------------------------
    # STEP 1: Build reverse index:
    #     node -> set(hubs)
    # This makes lookup O(1) instead of O(|H|) per trip
    # -------------------------------------------------------------

    origin_map = {}   # node -> hubs whose FORWARD reach includes node
    end_map    = {}   # node -> hubs whose REVERSE reach includes node

    # Build map for OC (forward distances)
    for hub, dist_dict in forward_distances.items():
        for node in dist_dict.keys():
            if node not in origin_map:
                origin_map[node] = set()
            origin_map[node].add(hub)

    # Build map for EC (reverse distances)
    for hub, dist_dict in backward_distances.items():
        for node in dist_dict.keys():
            if node not in end_map:
                end_map[node] = set()
            end_map[node].add(hub)

    # -------------------------------------------------------------
    # STEP 2: Build OC and EC for each trip in O(1) per trip
    # -------------------------------------------------------------
    OC = {}
    EC = {}

    i_count = 0
    for trip_id, (start_node, end_node) in trip_dict.items():
        i_count += 1
        if i_count % 1000 == 0:
            print(f"Computing OC/EC for trip {i_count} of {len(trip_dict)}")

        OC[trip_id] = origin_map.get(start_node, set())
        EC[trip_id] = end_map.get(end_node, set())

    return OC, EC

def count_fully_covered_trips(trip_ids, OC_data, EC_data, H):
    """
    trip_ids: list or set of trip IDs
    OC_data, EC_data: dict mapping trip_id -> set of nodes
    H: set of hub nodes
    """
    fully_covered_count = 0
    H=set(H)
    not_covered=[]
    for trip_id in trip_ids:
        OC_covered_nodes = set(OC_data[trip_id])  #find all hubs that partially cover this trip
        EC_covered_nodes = set(EC_data[trip_id])
        if OC_covered_nodes.intersection(H) and EC_covered_nodes.intersection(H):
            fully_covered_count += 1
        else:
            not_covered.append(trip_id)
    print(len(not_covered))
    return fully_covered_count

