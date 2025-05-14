import math

import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import GRB, Model, quicksum

import achref.src.logger as logger

logger = logger.get_logger(__name__)


def solve_refinery_optimization(
    crude_data, product_data, crude_product_yields, product_quality_reqs
):
    """
    Solves the Oil Refinery Optimization Problem.

    Args:
        crude_data (pd.DataFrame): DataFrame with columns ["Crude", "Cost", "Availability"]
        product_data (pd.DataFrame): DataFrame with columns ["Product", "Price", "Demand"]
        crude_product_yields (pd.DataFrame): DataFrame with columns ["Crude", "Product", "Yield", "Quality"]
        product_quality_reqs (pd.DataFrame): DataFrame with columns ["Product", "MinQuality"]

    Returns:
        tuple: (result_df, fig) where result_df contains the optimal solution and fig is a matplotlib figure
    """
    if not all(
        isinstance(df, pd.DataFrame)
        for df in [crude_data, product_data, crude_product_yields, product_quality_reqs]
    ):
        raise TypeError("All inputs must be pandas DataFrames")

    # Validate required columns
    if not set(["Crude", "Cost", "Availability"]).issubset(crude_data.columns):
        raise ValueError("crude_data must contain columns: Crude, Cost, Availability")
    if not set(["Product", "Price", "Demand"]).issubset(product_data.columns):
        raise ValueError("product_data must contain columns: Product, Price, Demand")
    if not set(["Crude", "Product", "Yield", "Quality"]).issubset(
        crude_product_yields.columns
    ):
        raise ValueError(
            "crude_product_yields must contain columns: Crude, Product, Yield, Quality"
        )
    if not set(["Product", "MinQuality"]).issubset(product_quality_reqs.columns):
        raise ValueError(
            "product_quality_reqs must contain columns: Product, MinQuality"
        )

    logger.info("Starting Gurobi model for Oil Refinery Optimization")

    # Create optimization model
    model = Model("OilRefineryOptimization")
    model.setParam("OutputFlag", 0)

    # Get unique crudes and products
    crudes = crude_data["Crude"].unique().tolist()
    products = product_data["Product"].unique().tolist()

    # Extract data
    costs = {row["Crude"]: row["Cost"] for _, row in crude_data.iterrows()}
    availability = {
        row["Crude"]: row["Availability"] for _, row in crude_data.iterrows()
    }
    prices = {row["Product"]: row["Price"] for _, row in product_data.iterrows()}
    demands = {row["Product"]: row["Demand"] for _, row in product_data.iterrows()}
    min_qualities = {
        row["Product"]: row["MinQuality"] for _, row in product_quality_reqs.iterrows()
    }

    # Create a dictionary for yields and qualities
    yields = {}
    qualities = {}
    for _, row in crude_product_yields.iterrows():
        crude, product = row["Crude"], row["Product"]
        yields[(crude, product)] = row["Yield"]
        qualities[(crude, product)] = row["Quality"]

    # Decision variables: amount of each crude oil used
    x = {crude: model.addVar(name=f"x_{crude}", lb=0) for crude in crudes}

    # Calculated variables: amount of each product produced from each crude
    prod_from_crude = {}
    for crude in crudes:
        for product in products:
            key = (crude, product)
            if key in yields:
                prod_from_crude[key] = yields[key] * x[crude]

    # Calculate total production per product
    total_production = {}
    for product in products:
        total_production[product] = quicksum(
            prod_from_crude.get((crude, product), 0) for crude in crudes
        )

    # Objective: Maximize profit (revenue - cost)
    revenue = quicksum(
        prices[product] * total_production[product] for product in products
    )
    cost = quicksum(costs[crude] * x[crude] for crude in crudes)
    model.setObjective(revenue - cost, GRB.MAXIMIZE)

    # Constraints:

    # 1. Crude availability constraints
    for crude in crudes:
        model.addConstr(x[crude] <= availability[crude], name=f"avail_{crude}")

    # 2. Demand satisfaction constraints
    for product in products:
        model.addConstr(
            total_production[product] >= demands[product], name=f"demand_{product}"
        )

    # 3. Quality constraints
    for product in products:
        if product in min_qualities:
            # Linearized quality constraint: sum(q_ij * y_ij * x_i) >= Q_j^min * sum(y_ij * x_i)
            quality_numerator = quicksum(
                qualities.get((crude, product), 0)
                * yields.get((crude, product), 0)
                * x[crude]
                for crude in crudes
                if (crude, product) in yields
            )
            model.addConstr(
                quality_numerator >= min_qualities[product] * total_production[product],
                name=f"quality_{product}",
            )

    # Solve the model
    model.optimize()

    # Check if optimal solution was found
    if model.status != GRB.OPTIMAL:
        raise ValueError("Failed to find an optimal solution")

    # Extract results
    crude_results = pd.DataFrame(
        {
            "Crude": crudes,
            "Amount Used": [x[crude].X for crude in crudes],
            "Cost": [x[crude].X * costs[crude] for crude in crudes],
        }
    )

    # Calculate production results
    prod_results = []
    for product in products:
        prod_amount = sum(
            prod_from_crude.get((crude, product), 0).getValue()
            for crude in crudes
            if (crude, product) in prod_from_crude
        )
        revenue = prod_amount * prices[product]
        # Calculate average quality
        quality_numerator = sum(
            qualities.get((crude, product), 0)
            * prod_from_crude.get((crude, product), 0).getValue()
            for crude in crudes
            if (crude, product) in prod_from_crude
        )
        avg_quality = quality_numerator / prod_amount if prod_amount > 0 else 0

        prod_results.append(
            {
                "Product": product,
                "Amount Produced": prod_amount,
                "Revenue": revenue,
                "Average Quality": avg_quality,
                "Min Quality Required": min_qualities.get(product, "N/A"),
            }
        )

    product_results_df = pd.DataFrame(prod_results)

    # Create visualizations
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Oil Refinery Optimization Results", fontsize=16, y=1.02)

    # Plot 1: Crude Oil Usage
    axs[0, 0].bar(crude_results["Crude"], crude_results["Amount Used"])
    axs[0, 0].set_title("Crude Oil Usage")
    axs[0, 0].set_ylabel("Amount (Liters)")
    plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha="right")

    # Plot 2: Product Production
    axs[0, 1].bar(product_results_df["Product"], product_results_df["Amount Produced"])
    axs[0, 1].set_title("Product Production")
    axs[0, 1].set_ylabel("Amount (Liters)")
    plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha="right")

    # Plot 3: Profit/Cost Breakdown
    revenue_total = product_results_df["Revenue"].sum()
    cost_total = crude_results["Cost"].sum()
    profit = revenue_total - cost_total
    axs[1, 0].bar(
        ["Revenue", "Cost", "Profit"],
        [revenue_total, cost_total, profit],
        color=["green", "red", "blue"],
    )
    axs[1, 0].set_title("Financial Summary")
    axs[1, 0].set_ylabel("Amount ($)")

    # Plot 4: Quality vs Requirements
    products = product_results_df["Product"]
    achieved_quality = product_results_df["Average Quality"]
    required_quality = pd.to_numeric(
        product_results_df["Min Quality Required"], errors="coerce"
    )

    x = range(len(products))
    width = 0.35
    axs[1, 1].bar(
        [i - width / 2 for i in x], achieved_quality, width, label="Achieved Quality"
    )
    axs[1, 1].bar(
        [i + width / 2 for i in x], required_quality, width, label="Required Quality"
    )
    axs[1, 1].set_title("Product Quality Analysis")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(products)
    axs[1, 1].set_ylabel("Quality")
    axs[1, 1].legend()
    plt.setp(axs[1, 1].get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    # Combine results for return
    combined_results = {
        "Crude Usage": crude_results,
        "Product Production": product_results_df,
        "Total Profit": profit,
    }

    # Convert to a results dataframe with all key information
    result_summary = pd.DataFrame(
        {
            "Category": ["Total Revenue", "Total Cost", "Total Profit"],
            "Value": [revenue_total, cost_total, profit],
        }
    )

    return product_results_df, fig


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
        for i in nodes
        for j in nodes
        if i != j
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
            math.hypot(
                coords[route[i]][0] - coords[route[i + 1]][0],
                coords[route[i]][1] - coords[route[i + 1]][1],
            )
            for i in range(len(route) - 1)
        )
        rows.append(
            {
                "Route": ridx,
                "Sequence": "→".join(str(n) for n in route),
                "Load": load,
                "Distance": dist,
            }
        )
    return pd.DataFrame(rows)


def _plot_plne(routes, routes_df, coords, customers, depot, K):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    ax.scatter(*zip(*[coords[i] for i in customers]), c="blue", label="Customers")
    ax.scatter(*coords[depot], c="red", s=100, label="Depot")
    colors = plt.cm.get_cmap("tab10", K)
    for ridx, route in enumerate(routes):
        pts = [coords[n] for n in route]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=colors(ridx), label=f"Route {ridx+1}")
    ax.set_title("Vehicle Routes")
    ax.legend(loc="upper right")
    ax2 = axs[1]
    bar_width = 0.35
    idx = range(len(routes_df))
    ax2.bar(idx, routes_df["Load"], bar_width, label="Load")
    ax2.bar(
        [i + bar_width for i in idx], routes_df["Distance"], bar_width, label="Distance"
    )
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
        m.addConstrs(
            (quicksum(x[i, j] for j in nodes if j != i) == 1 for i in customers),
            "leave",
        )
        m.addConstrs(
            (quicksum(x[i, j] for i in nodes if i != j) == 1 for j in customers),
            "enter",
        )
        m.addConstr(quicksum(x[depot, j] for j in customers) == K, "dep_out")
        m.addConstr(quicksum(x[i, depot] for i in customers) == K, "dep_in")
        m.addConstrs(
            (
                u[i] - u[j] + Q * x[i, j] <= Q - demand[j]
                for i in customers
                for j in customers
                if i != j
            ),
            name="mtz",
        )
        m.addConstr(u[depot] == 0, "depot_load")
        m.optimize()
        if m.status != GRB.OPTIMAL:
            raise ValueError("Gurobi failed to find an optimal solution.")
        sol = m.getAttr("x", x)
        routes = _reconstruct_routes(sol, depot, K)
        routes_df = _build_routes_df(routes, demand, coords, depot)
        fig = _plot_plne(routes, routes_df, coords, customers, depot, K)
        return routes_df, fig
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Error in solve_plne: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in solve_plne: {e}")


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
        for i in nodes
        for j in nodes
        if i != j
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
            math.hypot(
                coords[route[i]][0] - coords[route[i + 1]][0],
                coords[route[i]][1] - coords[route[i + 1]][1],
            )
            for i in range(len(route) - 1)
        )
        rows.append(
            {
                "Route": ridx,
                "Sequence": "→".join(str(n) for n in route),
                "Load": load,
                "Distance": dist,
            }
        )
    return pd.DataFrame(rows)


def _plot_plne(routes, routes_df, coords, customers, depot, K):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]
    ax.scatter(*zip(*[coords[i] for i in customers]), c="blue", label="Customers")
    ax.scatter(*coords[depot], c="red", s=100, label="Depot")
    colors = plt.cm.get_cmap("tab10", K)
    for ridx, route in enumerate(routes):
        pts = [coords[n] for n in route]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=colors(ridx), label=f"Route {ridx+1}")
    ax.set_title("Vehicle Routes")
    ax.legend(loc="upper right")
    ax2 = axs[1]
    bar_width = 0.35
    idx = range(len(routes_df))
    ax2.bar(idx, routes_df["Load"], bar_width, label="Load")
    ax2.bar(
        [i + bar_width for i in idx], routes_df["Distance"], bar_width, label="Distance"
    )
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
        m.addConstrs(
            (quicksum(x[i, j] for j in nodes if j != i) == 1 for i in customers),
            "leave",
        )
        m.addConstrs(
            (quicksum(x[i, j] for i in nodes if i != j) == 1 for j in customers),
            "enter",
        )
        m.addConstr(quicksum(x[depot, j] for j in customers) == K, "dep_out")
        m.addConstr(quicksum(x[i, depot] for i in customers) == K, "dep_in")
        m.addConstrs(
            (
                u[i] - u[j] + Q * x[i, j] <= Q - demand[j]
                for i in customers
                for j in customers
                if i != j
            ),
            name="mtz",
        )
        m.addConstr(u[depot] == 0, "depot_load")
        m.optimize()
        if m.status != GRB.OPTIMAL:
            raise ValueError("Gurobi failed to find an optimal solution.")
        sol = m.getAttr("x", x)
        routes = _reconstruct_routes(sol, depot, K)
        routes_df = _build_routes_df(routes, demand, coords, depot)
        fig = _plot_plne(routes, routes_df, coords, customers, depot, K)
        return routes_df, fig
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Error in solve_plne: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error in solve_plne: {e}")
