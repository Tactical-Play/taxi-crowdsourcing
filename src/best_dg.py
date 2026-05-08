from collections import defaultdict
import heapq

# 🔥 build reverse index once
def build_reverse_index(T, OC, EC):
    OC_rev = defaultdict(set)
    EC_rev = defaultdict(set)

    for tj in T:
        for s in OC[tj]:
            OC_rev[s].add(tj)
        for s in EC[tj]:
            EC_rev[s].add(tj)

    return OC_rev, EC_rev


def update_utilities_dg(T, s_theta, OC, EC, o, e, MU, delta, OC_rev, EC_rev):

    seen = set()

    # ---- ORIGIN SIDE ----
    for tj in OC_rev[s_theta]:
        seen.add(tj)

        oj_prev, ej_prev = o[tj], e[tj]
        oj_new, ej_new = oj_prev, ej_prev

        OC_tj = OC[tj]

        if s_theta in OC_tj:
            OC_tj.discard(s_theta)
            OC_rev[s_theta].discard(tj)

            l = len(OC_tj)

            if l == 0:
                oj_new = 0
            elif l == 1:
                replacement = next(iter(OC_tj))
                MU[replacement] += 1

        # ---- DESTINATION SIDE ----
        if s_theta in EC[tj]:
            EC_tj = EC[tj]
            EC_tj.discard(s_theta)
            EC_rev[s_theta].discard(tj)

            l = len(EC_tj)

            if l == 0:
                ej_new = 0
            elif l == 1:
                replacement = next(iter(EC_tj))
                MU[replacement] += 1

        # ---- UPDATE DELTA ----
        if oj_prev == 1 and ej_prev == 1:
            if oj_new == 0 and ej_new == 1:
                for si in EC[tj]:
                    delta[si] -= 1
                    MU[si] -= 1

            elif oj_new == 1 and ej_new == 0:
                for si in OC[tj]:
                    delta[si] -= 1
                    MU[si] -= 1

        o[tj], e[tj] = oj_new, ej_new

    # ---- Remaining EC-only trips ----
    for tj in EC_rev[s_theta]:
        if tj in seen:
            continue

        oj_prev, ej_prev = o[tj], e[tj]
        oj_new, ej_new = oj_prev, ej_prev

        EC_tj = EC[tj]

        if s_theta in EC_tj:
            EC_tj.discard(s_theta)
            EC_rev[s_theta].discard(tj)

            l = len(EC_tj)

            if l == 0:
                ej_new = 0
            elif l == 1:
                replacement = next(iter(EC_tj))
                MU[replacement] += 1

        if oj_prev == 1 and ej_prev == 1 and oj_new == 1 and ej_new == 0:
            for si in OC[tj]:
                delta[si] -= 1
                MU[si] -= 1

        o[tj], e[tj] = oj_new, ej_new


def initialize_dg(S, H0, T, OC, EC, o, e, MU, delta):
    for si in S.union(H0):
        MU[si] = 0
        delta[si] = 0

    for tj in T:
        o[tj], e[tj] = 1, 1

        if len(OC[tj]) == 1:
            si = next(iter(OC[tj]))
            MU[si] += 1

        if len(EC[tj]) == 1:
            si = next(iter(EC[tj]))
            MU[si] += 1

        for si in OC[tj]:
            delta[si] += 1

        for si in EC[tj]:
            delta[si] += 1


def decremental_greedy(S, k, H0, T, OC, EC, o, e, MU, delta):

    H = set(S).union(H0)

    OC_rev, EC_rev = build_reverse_index(T, OC, EC)
    initialize_dg(S, H0, T, OC, EC, o, e, MU, delta)

    # 🔥 HEAP initialization
    heap = []
    for s in H - H0:
        heapq.heappush(heap, (MU[s], delta[s], -s, s))

    removed = set()
    limit = len(H) - k

    for i in range(limit):
        if i % 500 == 0:
            print(f"Iteration no. {i} of {limit}")

        # 🔥 Lazy heap pop
        while True:
            mu_val, delta_val, neg_s, s_theta = heapq.heappop(heap)

            if s_theta in removed:
                continue

            if MU[s_theta] != mu_val or delta[s_theta] != delta_val:
                continue

            break

        H.remove(s_theta)
        removed.add(s_theta)

        update_utilities_dg(
            T, s_theta, OC, EC, o, e, MU, delta,
            OC_rev, EC_rev
        )

        # 🔥 push updated neighbors back into heap
        for s in H - H0:
            if s not in removed:
                heapq.heappush(heap, (MU[s], delta[s], -s, s))

    return H - H0