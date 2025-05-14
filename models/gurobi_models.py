from gurobipy import Model, GRB
import pandas as pd
import math
from gurobipy import quicksum
import matplotlib.pyplot as plt
import achref.src.logger as logger

logger = logger.get_logger(__name__)

def solve_pl(data, total_resource=100):
    required_cols = {"Product", "Profit/Unit", "Resource Usage"}
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Missing required columns in 'data': {missing}")
    if not isinstance(total_resource, (int, float)) or total_resource <= 0:
        raise ValueError("'total_resource' must be a positive number.")

    model = Model("ProductionPlanning")
    logger.info("Starting Gurobi model for production planning")
    logger.info("Data: %s", data)

    model.setParam("OutputFlag", 0)

    # Extract product info
    products = data["Product"].tolist()
    profits = data["Profit/Unit"].tolist()
    usage = data["Resource Usage"].tolist()

    # Create decision variables
    x = {
        prod: model.addVar(name=f"x_{prod}", lb=0, vtype=GRB.CONTINUOUS)
        for prod in products
    }

    # Objective: Maximise total profit
    model.setObjective(
        sum(profits[i] * x[products[i]] for i in range(len(products))),
        GRB.MAXIMIZE,
    )

    # Constraint: total resource usage <= available
    model.addConstr(
        sum(usage[i] * x[products[i]] for i in range(len(products))) <= total_resource,
        name="ResourceConstraint",
    )

    # Solve
    model.optimize()

    # Extract results
    result_df = pd.DataFrame({
        "Product": products,
        "Quantity Produced": [x[prod].X for prod in products],
        "Profit": [x[prod].X * profits[i] for i, prod in enumerate(products)]
    })
    
    result_df["Profit per Resource"] = result_df["Profit"] / data["Resource Usage"]

    # Prepare a single figure with 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Production Planning Visualisations", fontsize=16, y=1.02)

    # Plot 1: Quantity Produced
    axs[0, 0].bar(result_df["Product"], result_df["Quantity Produced"])
    axs[0, 0].set_title("Quantity Produced per Product")

    # Plot 2: Stacked Bar (Quantity + Profit)
    axs[0, 1].bar(result_df["Product"], result_df["Quantity Produced"], label="Quantity")
    axs[0, 1].bar(result_df["Product"], result_df["Profit"], 
                  bottom=result_df["Quantity Produced"], label="Profit", alpha=0.6)
    axs[0, 1].set_title("Stacked Bar: Quantity + Profit")
    axs[0, 1].legend()

    # Plot 3: Pie Chart of Profit
    axs[1, 0].pie(result_df["Profit"], labels=result_df["Product"], autopct='%1.1f%%')
    axs[1, 0].set_title("Profit Share per Product")

    # Plot 4: Cumulative Profit
    df_sorted = result_df.sort_values(by="Profit", ascending=False).reset_index(drop=True)
    df_sorted["Cumulative Profit"] = df_sorted["Profit"].cumsum()
    axs[1, 1].plot(df_sorted["Product"], df_sorted["Cumulative Profit"], marker="o")
    axs[1, 1].set_title("Cumulative Profit by Product")
    axs[1, 1].set_ylabel("Cumulative Profit")

    # Plot 5: Profit per Resource
    axs[2, 0].bar(result_df["Product"], result_df["Profit per Resource"])
    axs[2, 0].set_title("Profit per Unit of Resource Used")
    axs[2, 0].set_ylabel("Efficiency")

    # Remove unused subplot (bottom right)
    axs[2, 1].axis('off')

    fig.tight_layout()

    return result_df, fig


def _validate_plne_input(data, vehicle_capacity, num_vehicles):
    required_cols = {"Node", "X", "Y", "Demand"}
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise ValueError(f"Missing required columns in 'data': {missing}")
    if not isinstance(vehicle_capacity, (int, float)) or vehicle_capacity <= 0:
        raise ValueError("'vehicle_capacity' must be a positive number.")
    if not isinstance(num_vehicles, int) or num_vehicles <= 0:
        raise ValueError("'num_vehicles' must be a positive integer.")

def _prepare_plne_data(data):
    coords = {int(r.Node): (r.X, r.Y) for _, r in data.iterrows()}
    demand = {int(r.Node): r.Demand for _, r in data.iterrows()}
    nodes = list(coords.keys())
    depot = 0
    if depot not in nodes:
        raise ValueError("Depot node (0) is missing from input.")
    customers = [i for i in nodes if i != depot]
    return coords, demand, nodes, depot, customers

def _compute_distance_matrix(coords, nodes):
    return {
        (i, j): math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
        for i in nodes for j in nodes if i != j
    }

def _reconstruct_routes(sol, depot, K):
    starts = [j for (i, j), val in sol.items() if i == depot and val > 0.5]
    if len(starts) != K:
        raise ValueError(f"Expected {K} routes out of depot, got {len(starts)}")
    succ = {i: j for (i, j), val in sol.items() if i != depot and val > 0.5}
    routes = []
    for start in starts:
        route = [depot, start]
        cur = start
        while cur != depot:
            nxt = succ.get(cur)
            if nxt is None:
                raise ValueError(f"Incomplete route starting at node {start}.")
            route.append(nxt)
            cur = nxt
        routes.append(route)
    return routes

def _build_routes_df(routes, demand, coords, depot):
    rows = []
    for ridx, route in enumerate(routes, start=1):
        load = sum(demand[n] for n in route if n != depot)
        dist = sum(
            math.hypot(coords[route[i]][0] - coords[route[i+1]][0],
                       coords[route[i]][1] - coords[route[i+1]][1])
            for i in range(len(route)-1)
        )
        rows.append({
            "Route": ridx,
            "Sequence": "→".join(str(n) for n in route),
            "Load": load,
            "Distance": dist
        })
    return pd.DataFrame(rows)

def _plot_plne(routes, routes_df, coords, customers, depot, K):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    ax.scatter(*zip(*[coords[i] for i in customers]), c='blue', label='Customers')
    ax.scatter(*coords[depot], c='red', s=100, label='Depot')
    colors = plt.cm.get_cmap('tab10', K)
    for ridx, route in enumerate(routes):
        pts = [coords[n] for n in route]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, '-o', color=colors(ridx), label=f'Route {ridx+1}')
    ax.set_title("Vehicle Routes")
    ax.legend(loc='upper right')
    ax2 = axs[1]
    bar_width = 0.35
    idx = range(len(routes_df))
    ax2.bar(idx, routes_df["Load"], bar_width, label="Load")
    ax2.bar([i + bar_width for i in idx], routes_df["Distance"],
            bar_width, label="Distance")
    ax2.set_xticks([i + bar_width / 2 for i in idx])
    ax2.set_xticklabels([f"R{r}" for r in routes_df["Route"]])
    ax2.set_ylabel("Units / Distance")
    ax2.set_title("Load vs Distance per Route")
    ax2.legend()
    fig.tight_layout()
    return fig

def solve_plne(data: pd.DataFrame, vehicle_capacity: float, num_vehicles: int):
    """
    data: DataFrame with columns ["Node","X","Y","Demand"]
    vehicle_capacity: capacity Q of each vehicle
    num_vehicles: number of vehicles K
    Returns: (routes_df, fig)
      - routes_df: DataFrame with columns ["Route","Sequence","Load","Distance"]
      - fig: matplotlib.figure.Figure with the route‐map and summary bars
    """
    try:
        _validate_plne_input(data, vehicle_capacity, num_vehicles)
        coords, demand, nodes, depot, customers = _prepare_plne_data(data)
        Q, K = vehicle_capacity, num_vehicles
        cost = _compute_distance_matrix(coords, nodes)

        m = Model("CVRP")
        m.setParam("OutputFlag", 0)
        x = m.addVars(cost.keys(), vtype=GRB.BINARY, name="x")
        u = m.addVars(nodes, lb=0, ub=Q, vtype=GRB.CONTINUOUS, name="u")
        m.setObjective(quicksum(cost[i, j] * x[i, j] for i, j in cost), GRB.MINIMIZE)
        m.addConstrs((quicksum(x[i, j] for j in nodes if j != i) == 1 for i in customers), "leave")
        m.addConstrs((quicksum(x[i, j] for i in nodes if i != j) == 1 for j in customers), "enter")
        m.addConstr(quicksum(x[depot, j] for j in customers) == K, "dep_out")
        m.addConstr(quicksum(x[i, depot] for i in customers) == K, "dep_in")
        m.addConstrs(
            (u[i] - u[j] + Q * x[i, j] <= Q - demand[j]
             for i in customers for j in customers if i != j),
            name="mtz"
        )
        m.addConstr(u[depot] == 0, "depot_load")
        m.optimize()
        if m.status != GRB.OPTIMAL:
            raise ValueError("Gurobi failed to find an optimal solution.")
        sol = m.getAttr('x', x)
        routes = _reconstruct_routes(sol, depot, K)
        routes_df = _build_routes_df(routes, demand, coords, depot)
        fig = _plot_plne(routes, routes_df, coords, customers, depot, K)
        return routes_df, fig
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Error in solve_plne: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in solve_plne: {e}")