from collections import defaultdict

# 🔥 NEW: build reverse index once (call this before DG)
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

    # 🔥 NEW: only process affected trips
    affected_trips = OC_rev[s_theta] | EC_rev[s_theta]

    for tj in affected_trips:   # ⚡ CHANGE: instead of for tj in T

        oj_prev, ej_prev = o[tj], e[tj]
        oj_new, ej_new = oj_prev, ej_prev

        # ---- ORIGIN SIDE ----
        if s_theta in OC[tj]:

            OC[tj].discard(s_theta)   # ⚡ CHANGE: faster than set subtraction
            OC_rev[s_theta].discard(tj)  # 🔥 NEW: maintain reverse index

            l = len(OC[tj])  # ⚡ CHANGE: avoid repeated len()

            if l == 0:
                oj_new = 0
            elif l == 1:
                replacement = next(iter(OC[tj]))
                MU[replacement] = MU.get(replacement, 0) + 1  # ⚡ CHANGE: safe update

        # ---- DESTINATION SIDE ----
        if s_theta in EC[tj]:

            EC[tj].discard(s_theta)   # ⚡ CHANGE
            EC_rev[s_theta].discard(tj)  # 🔥 NEW

            l = len(EC[tj])  # ⚡ CHANGE

            if l == 0:
                ej_new = 0
            elif l == 1:
                replacement = next(iter(EC[tj]))
                MU[replacement] = MU.get(replacement, 0) + 1  # ⚡ CHANGE

        # ---- UPDATE DELTA ----
        if oj_prev == 1 and ej_prev == 1 and oj_new == 0 and ej_new == 1:
            for si in EC[tj]:
                delta[si] = delta.get(si, 0) - 1   # ⚡ CHANGE
                MU[si] = MU.get(si, 0) - 1

        if oj_prev == 1 and ej_prev == 1 and oj_new == 1 and ej_new == 0:
            for si in OC[tj]:
                delta[si] = delta.get(si, 0) - 1   # ⚡ CHANGE
                MU[si] = MU.get(si, 0) - 1

        # 🔥 OPTIONAL FIX: handle full loss case (0,0)
        if oj_prev == 1 and ej_prev == 1 and oj_new == 0 and ej_new == 0:
            # both sides lost → remove contribution
            pass  # (depends on your exact model)

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

    H = S.union(H0)

    # 🔥 NEW: build reverse index once
    OC_rev, EC_rev = build_reverse_index(T, OC, EC)

    initialize_dg(S, H0, T, OC, EC, o, e, MU, delta)

    limit = len(H) - k

    for i in range(limit):
        if i % 100 == 0:
            print(f"Iteration no. {i} of {limit}")

        candidates = list(H - H0)

        # ⚠️ FIX tie-break if s is string → adjust if needed
        s_theta = min(
            candidates,
            key=lambda s: (MU[s], delta[s], -s)   # ⚡ keep if s is int
        )

        H.remove(s_theta)

        # 🔥 NEW: pass reverse index
        update_utilities_dg(
            T, s_theta, OC, EC, o, e, MU, delta,
            OC_rev, EC_rev
        )

    return H - H0