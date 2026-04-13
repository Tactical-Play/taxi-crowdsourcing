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