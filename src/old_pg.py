import time

from itertools import combinations
from collections import defaultdict
import gc
# Global variables
# H = set()
# HH = set()
# o = {}
# e = {}
# U = defaultdict(int)
# Phi = defaultdict(int)

# Unordered Cartesian Product A x B (set of string/tuple)
def unordered_product(A, B):
    result = set()
    for a in A:
        for b in B:
            if a==b: result.add(a)
            else:
                if a<b:   result.add(tuple([a,b]))
                else:   result.add(tuple([b,a]))
    return result

# INITIALIZE-PG
def initialize_pg(S, H, HH, H0, T, OC, EC, o, e, U, Phi):
    #global o, e, U, Phi


    i=0
    for tj in T:
        i+=1
        if i%100==0 : print(f'Intitializing {i}th trip')
        o[tj] = 0
        e[tj] = 0

        NOC_tmp = OC[tj].union(unordered_product(OC[tj], S))
        NEC_tmp = EC[tj].union(unordered_product(EC[tj], S))
        X = NOC_tmp.union(NEC_tmp)
        Y = NOC_tmp.intersection(NEC_tmp)
        Z = X.difference(Y)
        for ss in Y:
            U[ss] = U[ss] + 1
        for ss in Z:
            Phi[ss] = Phi[ss] + 1
    
    if T:
        del NOC_tmp, NEC_tmp, X, Y, Z
    gc.collect()
    
    print("reached after reading all trips")
    for s in H0:#s0 should be a
        HH=HH.union({s},unordered_product(H, {s}))
        H.add(s)
        update_utilities_pg(
    T, HH, {s},
    OC, EC, S,
    o, e, U, Phi
)

# UPDATE-UTILITIES-PG
def update_utilities_pg(T, HH, H_theta, OC, EC, S, o, e, U, Phi):
    #global 

    for tj in T:
        oj, ej = o[tj], e[tj]
        oj_new, ej_new = oj, ej

        if H_theta.intersection(OC[tj]):
            oj_new = 1
        if H_theta.intersection(EC[tj]):
            ej_new = 1
        
        NOC_tmp = OC[tj].union(unordered_product(OC[tj], S))
        NEC_tmp = EC[tj].union(unordered_product(EC[tj], S))
        Y = NOC_tmp.intersection(NEC_tmp)
        X = NOC_tmp.difference(NEC_tmp)
        Z = NEC_tmp.difference(NOC_tmp)

        if oj == 0 and ej == 0 and oj_new == 1 and ej_new == 0:
            for ss in X.difference(HH):
                Phi[ss] -= 1
            for ss in Z.difference(HH):
                U[ss] += 1
                Phi[ss] -= 1
        
        elif oj == 0 and ej == 1 and oj_new == 1 and ej_new == 1:
            for ss in NOC_tmp.difference(HH):
                U[ss] -= 1

        elif oj == 0 and ej == 0 and oj_new == 0 and ej_new == 1:
            for ss in Z.difference(HH):
                Phi[ss] -= 1
            for ss in X.difference(HH):
                U[ss] += 1
                Phi[ss] -= 1

        elif oj == 1 and ej == 0 and oj_new == 1 and ej_new == 1:
            for ss in NEC_tmp.difference(HH):
                U[ss] -= 1
        
        elif oj == 0 and ej == 0 and oj_new == 1 and ej_new == 1:
            for ss in X.union(Z).difference(HH):
                Phi[ss] -= 1
            for ss in Y.difference(HH):
                U[ss] -= 1
        o[tj] = oj_new
        e[tj] = ej_new
    if T:
        del NOC_tmp, NEC_tmp, X, Y, Z
    gc.collect()

# PAIRWISE-GREEDY
import pickle

def save_pg_state(o, e, U, Phi, H, HH,filename="pg_init.pkl"):
    #global o, e, U, Phi, H, HH
    state = {
        "o": o,
        "e": e,
        "U": U,
        "Phi": Phi,
        "H": H,
        "HH": HH
    }
    with open(filename, "wb") as f:
        pickle.dump(state, f)
    print(f"Saved initialized state to {filename}")
def load_pg_state(filename="pg_init.pkl"):
    #global o, e, U, Phi, H, HH
    with open(filename, "rb") as f:
        state = pickle.load(f)
    return state
    # o = state["o"]
    # e = state["e"]
    # U = state["U"]
    # Phi = state["Phi"]
    # H = state["H"]
    # HH = state["HH"]
    #print(f"Loaded initialized state from {filename}")
def pairwise_greedy(
    S, k, H0, T, OC, EC,
    H, HH, o, e, U, Phi,
    use_cache=False,
    cache_file="pg_init.pkl"
):
    #global H, HH, o, e, U, Phi
    
    if not use_cache:
        initialize_pg(S, H, HH, H0, T, OC, EC, o, e, U, Phi)
    else:
        state = load_pg_state(cache_file)
        o = state["o"]
        e = state["e"]
        U = state["U"]
        Phi = state["Phi"]
        H = state["H"]
        HH = state["HH"]

    b = k
    while b > 0:
        print(f"iterations left = {b}")
        candidates = list(S - H)
        candidate_sets = unordered_product(candidates, candidates)
        candidate_sets = [cs for cs in candidate_sets if isinstance(cs,int) or len(cs) <= min(b, 2) ]
        #print(b,len(candidate_sets))
        if not candidate_sets:
            break
        max_U = max(U.get(cs, 0) for cs in candidate_sets)
        best_sets = [cs for cs in candidate_sets if U.get(cs, 0) == max_U]
        print(max_U,best_sets)
        if len(best_sets) > 1:
            max_phi = max(Phi.get(cs, 0) for cs in best_sets)
            best_sets = [cs for cs in best_sets if Phi.get(cs, 0) == max_phi]
            print(max_phi,best_sets)

        chosen = best_sets[0] if best_sets else set()
        print('chosen',chosen)
        H_theta = set()
        if isinstance(chosen,int) : H_theta.add(chosen)
        else:
            H_theta.add(chosen[0])
            H_theta.add(chosen[1])
        HH = HH.union(H_theta,unordered_product(H, H_theta))
        #print('HH - ',HH)
        H= H.union(H_theta)
        #print('H - ',H)
        update_utilities_pg(T, HH, H_theta, OC, EC, S, o, e, U, Phi)
        b -= len(H_theta)

    return H

def pairwise_greedy_multi(
    S, k_targets, H0, T, OC, EC, H, HH, o, e, U, Phi,
    cache_file="pg_init.pkl"
):

    #global H, HH, o, e, U, Phi

    k_targets = sorted(k_targets)

    # -------------------------------------------------
    # initialize once
    # -------------------------------------------------
    initialize_pg(S, H, HH, H0, T, OC, EC, o, e, U, Phi)

    save_pg_state(o, e, U, Phi, H, HH,cache_file)

    results = {}

    start_time = time.time()

    current_hubs = len(H)

    target_idx = 0

    while target_idx < len(k_targets):

        target_k = k_targets[target_idx]

        while len(H) < target_k:

            b = target_k - len(H)

            print(
                f"current hubs={len(H)} "
                f"target={target_k}"
            )

            candidates = list(S - H)

            candidate_sets = unordered_product(
                candidates,
                candidates
            )

            candidate_sets = [
                cs
                for cs in candidate_sets
                if isinstance(cs, int)
                or len(cs) <= min(b, 2)
            ]
            if not candidate_sets:
                break

            max_U = max(
                U.get(cs, 0)
                for cs in candidate_sets
            )

            best_sets = [
                cs
                for cs in candidate_sets
                if U.get(cs, 0) == max_U
            ]

            if len(best_sets) > 1:

                max_phi = max(
                    Phi.get(cs, 0)
                    for cs in best_sets
                )

                best_sets = [
                    cs
                    for cs in best_sets
                    if Phi.get(cs, 0) == max_phi
                ]

            chosen = best_sets[0]

            H_theta = set()

            if isinstance(chosen, int):
                H_theta.add(chosen)
            else:
                H_theta.add(chosen[0])
                H_theta.add(chosen[1])

            HH = HH.union(
                H_theta,
                unordered_product(H, H_theta)
            )

            H = H.union(H_theta)

            update_utilities_pg(T, HH, H_theta, OC, EC, S, o, e, U, Phi)

        results[target_k] = {
            "hubs": H.copy(),
            "elapsed": time.time() - start_time
        }

        print(
            f"Saved result for k={target_k}"
        )

        target_idx += 1

    return results