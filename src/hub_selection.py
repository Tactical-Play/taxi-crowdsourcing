def update_utilities_dg(T, s_theta, OC, EC, o, e, MU, delta):
    for tj in T:
        oj_prev, ej_prev = o[tj], e[tj]
        oj_new, ej_new = oj_prev, ej_prev

        if s_theta in OC[tj]:
            OC[tj] = OC[tj] - {s_theta}
            if len(OC[tj]) == 0:
                oj_new = 0
            elif len(OC[tj]) == 1:
                replacement = next(iter(OC[tj]))
                MU[replacement] += 1

        if s_theta in EC[tj]:
            EC[tj] = EC[tj] - {s_theta}
            if len(EC[tj]) == 0:
                ej_new = 0
            elif len(EC[tj]) == 1:
                replacement = next(iter(EC[tj]))
                MU[replacement] += 1

        if oj_prev == 1 and ej_prev == 1 and oj_new == 0 and ej_new == 1:
            for si in EC[tj]:
                delta[si] -= 1
                MU[si] -= 1

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
    H = S.union(H0)
    initialize_dg(S, H0, T, OC, EC, o, e, MU, delta)
    limit = len(H)-k
    for i in range(limit):
        if i%100==0: print(f"Iteration no. {i} of {limit}")
        # Select s_theta in H \ H0 with min MU, tie-break by min delta, then max index
        candidates = list(H - H0)
        s_theta = min(
            candidates,
            key=lambda s: (MU[s], delta[s], -s)  # s = 's0' → int(s[1:]) = 0

        )
        H.remove(s_theta)
        update_utilities_dg(T, s_theta, OC, EC, o, e, MU, delta)

    return H - H0

#o = {}
#e = {}
#MU = {}
#delta = {}
#final_hub_set = decremental_greedy(S=set(candidate_hubs), k=50, H0=set(), T=trip_dict,OC = OC_data.copy(), EC = EC_data.copy(),o= o,e= e,MU= MU,delta= delta)
#print("Final selected hubs:", final_hub_set)

##Incremental Greedy
# Global variables
#H = set() #Set of selected hub locations (initially empty,then includes existing hubs H0)
#o = {} # origin cover(yes/no) dict - {Tj:0, Tk:1,etc}
#e = {} # end cover
#to note: o and e indicate partial coverage of trip Tj w.r.t H
#so they each have as many entries as no. of trips
#and they represent the "present" set of hubs H
#U = {} # Utility of H union s, where s is a candidate hub.
#so entries are {s1:n1 , s2: n2, etc} ; n1,n2,... are no. of full covered trips
#Phi = {} #potential utility = trips fully covered but not partially covered


def incremental_greedy(S, k, H0, T, OC, EC):
    """
    Algorithm 1: INCREMENTAL-GREEDY
    Input: S (python set of candidate sites),
    k (number of hubs to select), H0 (python set of existing hubs),
           T (list of trips),
           OC (origin coverage dict i.e Tj : {set of sites}),
           EC (end coverage dict i.e Tj : {set of sites})
    Output: New hub locations H - H0
    """
    H = set()
    o = {}
    e = {}
    U = {}
    Phi = {}

    initialize_ig(S, H0, T, OC, EC, H, o, e, U, Phi)

    for _ in range(k):
        # Select s_theta with maximal U(s_theta), breaking ties with Phi(s_theta) and index
        candidates = [s for s in S if s not in H and s not in H0]
        if not candidates:
            break

        max_U = max(U[s] for s in candidates)
        best_s = [s for s in candidates if U[s] == max_U]

        if len(best_s) > 1:  #if multiple max utility sites
            max_Phi = max(Phi[s] for s in best_s)
            best_s = [s for s in best_s if Phi[s] == max_Phi]
            if len(best_s) > 1: # if still multiple sites
                best_s.sort()
                s_theta = best_s[0]
            else:
                s_theta = best_s[0]
        else:
            s_theta = best_s[0]

        H.add(s_theta)
        #del U[s_theta]
        #del Phi[s_theta]
        #it would be a good idea to delete utilities that won't be checked anymore

        # Update utilities
        update_utilities_ig(T, s_theta, OC, EC, H, o, e, U, Phi)

    return [s for s in H if s not in H0]


def initialize_ig(S, H0, T, OC, EC, H, o, e, U, Phi):
    """
    Algorithm 2: INITIALIZE-IG
    Input: S, H0, T, OC, EC
    Output: Mutates global H, o, e, U, Phi
    """

    # Initialize U and Phi
    for s in S.union(H0):
        U[s] = 0
        Phi[s] = 0

    for tj in T:
        o[tj] = 0
        e[tj] = 0
        for s in OC[tj]:
            Phi[s] += 1
        for s in EC[tj]:
            Phi[s] += 1
    #print('initial values: u - ',U,'\nphi - ',Phi,'\n')
    # Add existing hubs and update utilities
    for s in H0:
        H.add(s)
        update_utilities_ig(T, s, OC, EC)


def update_utilities_ig(T, s_theta, OC, EC, H, o, e, U, Phi):
    """
    Algorithm 3: UPDATE-UTILITIES-IG
    Input: T, s_theta, OC, EC
    Output: Mutates global o, e, U, Phi
    """

    for tj in T:
        o_prev = o[tj]
        e_prev = e[tj]
        o_new , e_new = o_prev , e_prev
        if s_theta in OC[tj]:
            o_new = 1
        if s_theta in EC[tj]:
            e_new = 1

        # Case 1: o=0, e=0 → o'=1, e'=0
        if o_prev == 0 and e_prev == 0:
            if o_new == 1 and e_new == 0:
                for s in EC[tj]:
                    if s not in H:
                        U[s] += 1
                        Phi[s] -= 1
                for s in OC[tj]:
                    if s not in H:
                        Phi[s] -= 1
            # Case 2: o=0, e=0 → o'=0, e'=1
            elif o_new == 0 and e_new == 1:
                for s in OC[tj]:
                    if s not in H:
                        U[s] += 1
                        Phi[s] -= 1
                for s in EC[tj]:
                    if s not in H:
                        Phi[s] -= 1

        # Case 3: o=1, e=0 → o'=1, e'=1
        elif o_prev == 1 and e_prev == 0:
            if o_new == 1 and e_new == 1:
                for s in EC[tj]:
                    if s not in H:
                        U[s] -= 1

        # Case 4: o=0, e=1 → o'=1, e'=1
        elif o_prev == 0 and e_prev == 1:
            if o_new == 1 and e_new == 1:
                for s in OC[tj]:
                    if s not in H:
                        U[s] -= 1

        # Update o and e
        o[tj] = o_new
        e[tj] = e_new
    #print('after including ',s_theta,' u - ',U,'\nphi - ',Phi,'\n')

# Reset globals before each run
#H = set()
#o = {}; e = {}; U = {}; Phi = {}
