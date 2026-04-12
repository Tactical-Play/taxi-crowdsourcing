from collections import defaultdict
from itertools import combinations

# -------------------------------
# PRECOMPUTATION
# -------------------------------

def build_reverse_index(T, OC, EC):
    OC_rev = defaultdict(set)
    EC_rev = defaultdict(set)

    for tj in T:
        for s in OC[tj]:
            OC_rev[s].add(tj)
        for s in EC[tj]:
            EC_rev[s].add(tj)

    return OC_rev, EC_rev


def precompute_pair_trips(S, OC_rev, EC_rev):
    """
    pair_to_trips[(s1,s2)] = affected trips
    """
    pair_to_trips = {}

    S_list = list(S)

    # singles
    for s in S_list:
        pair_to_trips[(s,)] = OC_rev[s] | EC_rev[s]

    # pairs
    for s1, s2 in combinations(S_list, 2):
        pair_to_trips[(s1, s2)] = (
            OC_rev[s1] | EC_rev[s1] |
            OC_rev[s2] | EC_rev[s2]
        )

    return pair_to_trips


# -------------------------------
# INITIALIZE U (PAIR GAINS)
# -------------------------------

def initialize_pair_gain(pair_to_trips, o, e):
    U = {}

    for pair, trips in pair_to_trips.items():

        gain = 0

        for tj in trips:
            if o[tj] == 0 and e[tj] == 0:
                gain += 1

        U[pair] = gain

    return U


# -------------------------------
# UPDATE AFTER SELECTING HUBS
# -------------------------------

def update_after_selection(H_theta, pair_to_trips, U, o, e, OC_rev, EC_rev):

    # 🔥 update o,e first
    for s in H_theta:
        for tj in OC_rev[s]:
            o[tj] = 1
        for tj in EC_rev[s]:
            e[tj] = 1

    # 🔥 recompute U ONLY for affected pairs
    affected_pairs = []

    for s in H_theta:
        for pair in pair_to_trips:
            if s in pair:
                affected_pairs.append(pair)

    affected_pairs = set(affected_pairs)

    for pair in affected_pairs:
        trips = pair_to_trips[pair]

        gain = 0
        for tj in trips:
            if o[tj] == 0 and e[tj] == 0:
                gain += 1

        U[pair] = gain


# -------------------------------
# MAIN ALGO
# -------------------------------

def pairwise_greedy(S, k, H0, T, OC, EC):

    H = set(H0)

    # 🔥 reverse index
    OC_rev, EC_rev = build_reverse_index(T, OC, EC)

    # 🔥 initialize state
    max_tj = max(T) + 1
    o = [0] * max_tj
    e = [0] * max_tj

    # apply H0
    for s in H0:
        for tj in OC_rev[s]:
            o[tj] = 1
        for tj in EC_rev[s]:
            e[tj] = 1

    print("Precomputing pair → trips (heavy but one-time)...")
    pair_to_trips = precompute_pair_trips(S, OC_rev, EC_rev)

    print("Initializing pair gains...")
    U = initialize_pair_gain(pair_to_trips, o, e)

    b = k

    while b > 0:

        print(f"iterations left = {b}")

        # 🔥 filter valid candidates
        candidates = [
            p for p in U.keys()
            if all(s not in H for s in p) and len(p) <= b
        ]

        if not candidates:
            break

        # 🔥 best pair selection
        best_pair = max(candidates, key=lambda p: U[p])

        print("chosen:", best_pair, "gain:", U[best_pair])

        H_theta = set(best_pair)

        # 🔥 update
        update_after_selection(
            H_theta, pair_to_trips, U, o, e, OC_rev, EC_rev
        )

        H |= H_theta
        b -= len(H_theta)

    return H