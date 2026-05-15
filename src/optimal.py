import gurobipy as gp
from gurobipy import GRB

def solve_gurobi(S, H0, T, OC, EC, k, timelimit=None, verbose=True):
    """
    Build and solve the hub-selection ILP using gurobipy.

    Arguments:
      S    : iterable of candidate site IDs (can be strings or ints)
      H0   : iterable of existing hub site IDs
      T    : iterable of trip IDs
      OC   : dict mapping trip -> set of site IDs (OC(Tj))
      EC   : dict mapping trip -> set of site IDs (EC(Tj))
      k    : int (number of new hubs to choose)
      timelimit : seconds (optional) to pass to Gurobi
      verbose : show solver output

    Returns:
      dict with solution: {'status': status, 'objval': value, 'x': {...}, 'o': {...}, 'e': {...}, 'U': {...}, 'model': model}
    """
    Sites = set(S) | set(H0)

    try:
        m = gp.Model("HubSelection_OPT")
        m.setParam("OutputFlag", 1 if verbose else 0)
        if timelimit is not None:
            m.setParam("TimeLimit", timelimit)

        # Variables
        x = {i: m.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in Sites}
        o = {j: m.addVar(vtype=GRB.BINARY, name=f"o_{j}") for j in T}
        e = {j: m.addVar(vtype=GRB.BINARY, name=f"e_{j}") for j in T}
        U = {j: m.addVar(vtype=GRB.BINARY, name=f"U_{j}") for j in T}

        m.update()

        # Objective: maximize sum_j U_j
        m.setObjective(gp.quicksum(U[j] for j in T), GRB.MAXIMIZE)

        # Constraints:
        # U_j <= o_j  and  U_j <= e_j
        for j in T:
            m.addConstr(U[j] <= o[j], name=f"U_le_o_{j}")
            m.addConstr(U[j] <= e[j], name=f"U_le_e_{j}")

        # o_j <= sum_{s in OC(Tj)} x_s
        # e_j <= sum_{s in EC(Tj)} x_s
        for j in T:
            oc_set = set(OC.get(j, ()))
            ec_set = set(EC.get(j, ()))

            if oc_set:
                m.addConstr(o[j] <= gp.quicksum(x[s] for s in oc_set), name=f"o_sumOC_{j}")
            
            if ec_set:
                m.addConstr(e[j] <= gp.quicksum(x[s] for s in ec_set), name=f"e_sumEC_{j}")
            
        # Sum x_i over Sites == k + |H0|
        m.addConstr(gp.quicksum(x[i] for i in Sites) == k + len(H0), name="Total_hubs_count")

        # x_i = 1 for all i in H0
        for i in H0:
            if i in x:
                m.addConstr(x[i] == 1, name=f"existing_hub_{i}")

        # Optimize
        m.optimize()
        status = m.Status
        objval = None
        if status == GRB.OPTIMAL or status == GRB.TIME_LIMIT or status == GRB.SUBOPTIMAL:
            # collect solution values (if variable has no value, .X raises; guard with .X if available)
            sol_x = {i: int(x[i].X) if x[i].X is not None else 0 for i in Sites}
            sol_o = {j: int(o[j].X) if o[j].X is not None else 0 for j in T}
            sol_e = {j: int(e[j].X) if e[j].X is not None else 0 for j in T}
            sol_U = {j: int(U[j].X) if U[j].X is not None else 0 for j in T}
            selected_hubs = {i for i, val in sol_x.items() if val == 1}
            try:
                objval = m.ObjVal
            except Exception:
                objval = None

            return {
                'status': status,
                'hubs': selected_hubs,
                'objval': objval,
                'x': sol_x,
                'o': sol_o,
                'e': sol_e,
                'U': sol_U,
                'model': m
            }
        else:
            return {'status': status, 'objval': None, 'x': {}, 'o': {}, 'e': {}, 'U': {}, 'model': m}

    except gp.GurobiError as err:
        print(f"Gurobi error: {err}")
        raise

import json
import gurobipy as gp
from gurobipy import GRB


def solve_for_gap_thresholds(
    S, H0, T, OC, EC, k,
    gaps=None,
    verbose=False,
    save_file="hub_results.jsonl"
):
    """
    Run Gurobi multiple times with different MIPGap values (descending),
    using warm-start from previous solutions.

    Saves each iteration immediately to disk.

    Returns:
        dict: gap -> solution dict
    """

    if gaps is None:
        gaps = [i / 100 for i in range(10, 0, -1)]  # 0.10 → 0.01

    results = {}
    prev_x = None

    # Clear/create output file
    open(save_file, "w").close()

    for gap in gaps:

        print(f"\n==============================")
        print(f"Solving with MIPGap = {gap*100:.0f}%")
        print(f"==============================")

        res = solve_gurobi_with_gap(
            S, H0, T, OC, EC, k,
            mipgap=gap,
            warm_start=prev_x,
            verbose=verbose
        )

        results[gap] = res

        # Update warm-start
        prev_x = res.get("x", None)

        # Prepare serializable record
        record = {
            "gap": gap,
            "status": int(res["status"]),
            "objective": res["objval"],
            "num_hubs": len(res["hubs"]),
            "hubs": sorted(list(res["hubs"]))
        }

        # Save immediately (append mode)
        with open(save_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Saved results for gap={gap:.2f}")

        # Quick summary
        print(f"Status: {res['status']}")
        print(f"Objective: {res['objval']}")
        print(f"#Hubs: {len(res['hubs'])}")

    return results


def solve_gurobi_with_gap(
    S, H0, T, OC, EC, k,
    mipgap=0.01,
    warm_start=None,
    verbose=True
):

    Sites = set(S) | set(H0)

    try:

        m = gp.Model("HubSelection_OPT")

        m.setParam("OutputFlag", 1 if verbose else 0)
        m.setParam("MIPGap", mipgap)

        # Variables
        x = {
            i: m.addVar(vtype=GRB.BINARY, name=f"x_{i}")
            for i in Sites
        }

        o = {
            j: m.addVar(vtype=GRB.BINARY, name=f"o_{j}")
            for j in T
        }

        e = {
            j: m.addVar(vtype=GRB.BINARY, name=f"e_{j}")
            for j in T
        }

        U = {
            j: m.addVar(vtype=GRB.BINARY, name=f"U_{j}")
            for j in T
        }

        m.update()

        # Warm-start
        if warm_start is not None:
            for i in x:
                if i in warm_start:
                    x[i].start = warm_start[i]

        # Objective
        m.setObjective(
            gp.quicksum(U[j] for j in T),
            GRB.MAXIMIZE
        )

        # Constraints
        for j in T:
            m.addConstr(U[j] <= o[j])
            m.addConstr(U[j] <= e[j])

        for j in T:

            oc_set = set(OC.get(j, ()))
            ec_set = set(EC.get(j, ()))

            if oc_set:
                m.addConstr(
                    o[j] <= gp.quicksum(x[s] for s in oc_set)
                )

            if ec_set:
                m.addConstr(
                    e[j] <= gp.quicksum(x[s] for s in ec_set)
                )

        m.addConstr(
            gp.quicksum(x[i] for i in Sites)
            == k + len(H0)
        )

        for i in H0:
            if i in x:
                m.addConstr(x[i] == 1)

        # Optimize
        m.optimize()

        status = m.Status

        if status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:

            sol_x = {
                i: int(x[i].X)
                for i in Sites
            }

            selected_hubs = {
                i for i, val in sol_x.items()
                if val == 1
            }

            try:
                objval = m.ObjVal
            except:
                objval = None

            return {
                "status": status,
                "hubs": selected_hubs,
                "objval": objval,
                "x": sol_x,
                "model": m
            }

        else:

            return {
                "status": status,
                "hubs": set(),
                "objval": None,
                "x": {},
                "model": m
            }

    except gp.GurobiError as err:
        print(f"Gurobi error: {err}")
        raise
def solve_gurobi_with_gap(S, H0, T, OC, EC, k, mipgap=0.01, warm_start=None, verbose=True):
    Sites = set(S) | set(H0)

    try:
        m = gp.Model("HubSelection_OPT")
        m.setParam("OutputFlag", 1 if verbose else 0)
        m.setParam("MIPGap", mipgap)

        # Variables
        x = {i: m.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in Sites}
        o = {j: m.addVar(vtype=GRB.BINARY, name=f"o_{j}") for j in T}
        e = {j: m.addVar(vtype=GRB.BINARY, name=f"e_{j}") for j in T}
        U = {j: m.addVar(vtype=GRB.BINARY, name=f"U_{j}") for j in T}

        m.update()

        if warm_start is not None:
            for i in x:
                if i in warm_start:
                    x[i].start = warm_start[i]

        # Objective
        m.setObjective(gp.quicksum(U[j] for j in T), GRB.MAXIMIZE)

        # Constraints
        for j in T:
            m.addConstr(U[j] <= o[j])
            m.addConstr(U[j] <= e[j])

        for j in T:
            oc_set = set(OC.get(j, ()))
            ec_set = set(EC.get(j, ()))

            if oc_set:
                m.addConstr(o[j] <= gp.quicksum(x[s] for s in oc_set))

            if ec_set:
                m.addConstr(e[j] <= gp.quicksum(x[s] for s in ec_set))

        m.addConstr(gp.quicksum(x[i] for i in Sites) == k + len(H0))

        for i in H0:
            if i in x:
                m.addConstr(x[i] == 1)

        # Optimize
        m.optimize()

        status = m.Status

        if status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            sol_x = {i: int(x[i].X) for i in Sites}
            selected_hubs = {i for i, val in sol_x.items() if val == 1}

            try:
                objval = m.ObjVal
            except:
                objval = None

            return {
                'status': status,
                'hubs': selected_hubs,
                'objval': objval,
                'x': sol_x,
                'model': m
            }

        else:
            return {
                'status': status,
                'hubs': set(),
                'objval': None,
                'x': {},
                'model': m
            }

    except gp.GurobiError as err:
        print(f"Gurobi error: {err}")
        raise  
