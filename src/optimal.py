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




import os
import json
import gurobipy as gp
from gurobipy import GRB


def _save_solution(model, x, gap, checkpoint_file, results_file, mst_dir):
    """
    Save current incumbent immediately.
    """

    sol_x = {
        i: int(round(x[i].X))
        for i in x
    }

    selected_hubs = sorted([
        i for i, val in sol_x.items()
        if val == 1
    ])

    objval = model.ObjVal
    objbound = model.ObjBound

    record = {
        "gap": gap,
        "objective": objval,
        "best_bound": objbound,
        "num_hubs": len(selected_hubs),
        "hubs": selected_hubs
    }

    # append durable log
    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    # overwrite checkpoint
    checkpoint = {
        "last_gap": gap,
        "objective": objval,
        "best_bound": objbound,
        "x": {
            str(i): sol_x[i]
            for i in sol_x
        }
    }

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)

    # save mst
    if mst_dir is not None:
        mst_file = os.path.join(
            mst_dir,
            f"gap_{int(gap * 100)}.mst"
        )
        model.write(mst_file)

    print(
        f"Saved {gap*100:.0f}% | "
        f"Obj={objval:.2f} | "
        f"Bound={objbound:.2f} | "
        f"Hubs={len(selected_hubs)}"
    )

    return selected_hubs


def solve_gurobi_checkpointed(
    S,
    H0,
    T,
    OC,
    EC,
    k,
    verbose=True,
    checkpoint_file="gurobi_checkpoint.json",
    results_file="hub_results.jsonl",
    mst_dir="mst_checkpoints",
    thresholds=None
):
    """
    Sequential continuation solve:
    10% -> 9% -> ... -> 1%
    using SAME model state.
    """

    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 0, -1)]

    Sites = set(S) | set(H0)

    os.makedirs(mst_dir, exist_ok=True)

    completed_thresholds = set()

    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    completed_thresholds.add(rec["gap"])
        except:
            pass

    thresholds = [
        th for th in thresholds
        if th not in completed_thresholds
    ]

    print("Remaining thresholds:", thresholds)

    try:
        m = gp.Model("HubSelection_OPT")
        m.setParam("OutputFlag", 1 if verbose else 0)

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

        # Resume checkpoint warm start
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r") as f:
                    chk = json.load(f)

                warm = chk.get("x", {})

                print("Applying checkpoint warm-start...")

                for i in x:
                    key = str(i)
                    if key in warm:
                        x[i].Start = warm[key]

            except Exception as err:
                print(f"Checkpoint load failed: {err}")

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

        if not os.path.exists(results_file):
            open(results_file, "w").close()

        # CONTINUATION SOLVE
        for gap in thresholds:

            print(f"\n=== Solving to {gap*100:.0f}% gap ===")

            m.Params.MIPGap = gap
            m.optimize()

            if m.SolCount == 0:
                print(f"No feasible solution at {gap*100:.0f}%")
                continue

            _save_solution(
                m,
                x,
                gap,
                checkpoint_file,
                results_file,
                mst_dir
            )

        status = m.Status

        if m.SolCount > 0:
            sol_x = {
                i: int(round(x[i].X))
                for i in Sites
            }

            selected_hubs = {
                i for i, val in sol_x.items()
                if val == 1
            }

            return {
                "status": status,
                "hubs": selected_hubs,
                "objval": m.ObjVal,
                "x": sol_x,
                "model": m
            }

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