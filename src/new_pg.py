from collections import defaultdict
from itertools import combinations

def build_reverse_index(T, OC, EC):
    OC_rev = defaultdict(set)
    EC_rev = defaultdict(set)

    for tj in T:
        for s in OC[tj]:
            OC_rev[s].add(tj)
        for s in EC[tj]:
            EC_rev[s].add(tj)

    return OC_rev, EC_rev


def compute_pair_gain(s1, s2, T, OC_rev, EC_rev, o, e):
    """
    Compute gain of adding {s1, s2}
    Only over affected trips
    """

    affected = (
        OC_rev[s1] | EC_rev[s1] |
        OC_rev[s2] | EC_rev[s2]
    )

    gain = 0

    for tj in affected:
        oj, ej = o[tj], e[tj]

        oj_new = oj or (s1 in OC_rev and tj in OC_rev[s1]) or (s2 in OC_rev and tj in OC_rev[s2])
        ej_new = ej or (s1 in EC_rev and tj in EC_rev[s1]) or (s2 in EC_rev and tj in EC_rev[s2])

        # simple gain condition (adjust if needed)
        if oj == 0 and ej == 0 and (oj_new or ej_new):
            gain += 1

    return gain


def update_state(H_theta, OC_rev, EC_rev, o, e):
    """
    Update o,e using reverse index
    """

    for s in H_theta:

        for tj in OC_rev[s]:
            o[tj] = 1

        for tj in EC_rev[s]:
            e[tj] = 1


def pairwise_greedy(S, k, H0, T, OC, EC):

    H = set(H0)

    # 🔥 reverse index
    OC_rev, EC_rev = build_reverse_index(T, OC, EC)

    # 🔥 initialize state
    max_tj = max(T) + 1
    o = [0] * max_tj
    e = [0] * max_tj

    # apply H0
    update_state(H0, OC_rev, EC_rev, o, e)

    # 🔥 precompute candidate pairs ONCE
    S_list = list(S)
    all_pairs = [(s,) for s in S_list] + list(combinations(S_list, 2))

    b = k

    while b > 0:

        print(f"iterations left = {b}")

        best_gain = -1
        best_choice = None

        # 🔥 iterate candidates lazily
        for cand in all_pairs:

            if any(s in H for s in cand):
                continue

            if len(cand) > b:
                continue

            if len(cand) == 1:
                gain = compute_pair_gain(cand[0], cand[0], T, OC_rev, EC_rev, o, e)
            else:
                gain = compute_pair_gain(cand[0], cand[1], T, OC_rev, EC_rev, o, e)

            if gain > best_gain:
                best_gain = gain
                best_choice = cand

        if best_choice is None:
            break

        print("chosen:", best_choice, "gain:", best_gain)

        H_theta = set(best_choice)

        # 🔥 update state efficiently
        update_state(H_theta, OC_rev, EC_rev, o, e)

        H |= H_theta
        b -= len(H_theta)

    return H