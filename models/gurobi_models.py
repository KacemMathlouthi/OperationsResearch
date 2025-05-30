import math

import matplotlib.pyplot as plt
import pandas as pd
from gurobipy import GRB, Model, quicksum

import achref.src.logger as logger

logger = logger.get_logger(__name__)


def solve_diet_problem(foods_df, requirements_df):
    """
    Solves the Diet Problem using Linear Programming with flexible number of foods and nutrients.

    Args:
        foods_df (pd.DataFrame): DataFrame with columns ['Food', 'Cost'] + nutrient columns
        requirements_df (pd.DataFrame): DataFrame with columns ['Nutrient', 'Minimum']

    Returns:
        tuple: (result_df, fig) where result_df contains the optimal solution and fig is a matplotlib figure

    Raises:
        TypeError: If inputs are not DataFrames or contain invalid data types
        ValueError: If data validation fails or optimization problem cannot be solved
        Exception: If Gurobi solver encounters an error
    """
    logger.info("Starting Gurobi model for Diet Problem")

    try:
        # Type validation
        if not isinstance(foods_df, pd.DataFrame):
            raise TypeError("foods_df must be a pandas DataFrame")
        if not isinstance(requirements_df, pd.DataFrame):
            raise TypeError("requirements_df must be a pandas DataFrame")

        # Empty data validation
        if foods_df.empty:
            raise ValueError(
                "Foods data cannot be empty. Please provide at least one food item."
            )
        if requirements_df.empty:
            raise ValueError(
                "Requirements data cannot be empty. Please provide at least one nutritional requirement."
            )

        # Required columns validation
        required_food_cols = {"Food", "Cost"}
        if not required_food_cols.issubset(foods_df.columns):
            missing = required_food_cols - set(foods_df.columns)
            raise ValueError(
                f"Missing required columns in foods data: {missing}. Required columns: {required_food_cols}"
            )

        required_req_cols = {"Nutrient", "Minimum"}
        if not required_req_cols.issubset(requirements_df.columns):
            missing = required_req_cols - set(requirements_df.columns)
            raise ValueError(
                f"Missing required columns in requirements data: {missing}. Required columns: {required_req_cols}"
            )

        # Duplicate validation
        if foods_df["Food"].duplicated().any():
            duplicates = foods_df[foods_df["Food"].duplicated()]["Food"].tolist()
            raise ValueError(
                f"Duplicate food names found: {duplicates}. Each food must have a unique name."
            )

        if requirements_df["Nutrient"].duplicated().any():
            duplicates = requirements_df[requirements_df["Nutrient"].duplicated()][
                "Nutrient"
            ].tolist()
            raise ValueError(
                f"Duplicate nutrient names found: {duplicates}. Each nutrient must be unique."
            )

        # Get nutrient columns (all columns except 'Food' and 'Cost')
        nutrient_cols = [col for col in foods_df.columns if col not in ["Food", "Cost"]]

        if not nutrient_cols:
            raise ValueError(
                "No nutrient columns found in foods data. Please include at least one nutrient column (e.g., 'Protein', 'Fat', 'Carbs')."
            )

        # Data type validation
        try:
            foods_df["Cost"] = pd.to_numeric(foods_df["Cost"], errors="raise")
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cost column contains non-numeric values. All costs must be numbers. Error: {str(e)}"
            )

        try:
            requirements_df["Minimum"] = pd.to_numeric(
                requirements_df["Minimum"], errors="raise"
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Minimum column contains non-numeric values. All requirements must be numbers. Error: {str(e)}"
            )

        for col in nutrient_cols:
            try:
                foods_df[col] = pd.to_numeric(foods_df[col], errors="raise")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Nutrient column '{col}' contains non-numeric values. All nutrient values must be numbers. Error: {str(e)}"
                )

        # Value range validation
        if (foods_df["Cost"] < 0).any():
            negative_costs = foods_df[foods_df["Cost"] < 0]["Food"].tolist()
            raise ValueError(
                f"Negative costs found for foods: {negative_costs}. All costs must be non-negative."
            )

        if (requirements_df["Minimum"] <= 0).any():
            non_positive_reqs = requirements_df[requirements_df["Minimum"] <= 0][
                "Nutrient"
            ].tolist()
            raise ValueError(
                f"Non-positive requirements found for nutrients: {non_positive_reqs}. All requirements must be positive."
            )

        for col in nutrient_cols:
            if (foods_df[col] < 0).any():
                negative_nutrients = foods_df[foods_df[col] < 0]["Food"].tolist()
                raise ValueError(
                    f"Negative {col} values found for foods: {negative_nutrients}. All nutrient values must be non-negative."
                )

        # Check if any required nutrient is missing from foods data
        missing_nutrients = set(requirements_df["Nutrient"]) - set(nutrient_cols)
        if missing_nutrients:
            raise ValueError(
                f"Required nutrients missing from foods data: {missing_nutrients}. Please add these nutrient columns to your foods data."
            )

        # Check for zero costs (potential unbounded solution)
        if (foods_df["Cost"] == 0).any():
            zero_cost_foods = foods_df[foods_df["Cost"] == 0]["Food"].tolist()
            logger.warning(
                f"Foods with zero cost detected: {zero_cost_foods}. This may lead to unrealistic solutions."
            )

        # Feasibility pre-check: ensure at least one food provides each required nutrient
        for _, req_row in requirements_df.iterrows():
            nutrient = req_row["Nutrient"]
            if nutrient in nutrient_cols:
                max_nutrient = foods_df[nutrient].max()
                if max_nutrient == 0:
                    raise ValueError(
                        f"No food provides the required nutrient '{nutrient}'. Problem is infeasible."
                    )
                if max_nutrient < req_row["Minimum"]:
                    logger.warning(
                        f"Maximum {nutrient} content ({max_nutrient}) is less than requirement ({req_row['Minimum']}). May need multiple foods."
                    )

    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

    # Create optimization model
    try:
        model = Model("DietProblem")
        model.setParam("OutputFlag", 0)

        # Set time limit to prevent infinite solving (60 seconds)
        model.setParam("TimeLimit", 60)

        logger.info("Creating decision variables...")

        # Decision variables: units of each food
        food_vars = {}
        for i, food_name in enumerate(foods_df["Food"]):
            if pd.isna(food_name) or str(food_name).strip() == "":
                raise ValueError(
                    f"Empty or invalid food name at row {i}. All foods must have valid names."
                )
            food_vars[food_name] = model.addVar(name=f"Food_{i}_{food_name}", lb=0)

        logger.info("Setting objective function...")

        # Objective: Minimize total cost
        model.setObjective(
            quicksum(
                foods_df.loc[foods_df["Food"] == food, "Cost"].iloc[0] * var
                for food, var in food_vars.items()
            ),
            GRB.MINIMIZE,
        )

        logger.info("Adding constraints...")

        # Constraints: Nutritional requirements
        for _, req_row in requirements_df.iterrows():
            nutrient = req_row["Nutrient"]
            min_requirement = req_row["Minimum"]

            if pd.isna(nutrient) or str(nutrient).strip() == "":
                raise ValueError(
                    "Empty or invalid nutrient name found. All nutrients must have valid names."
                )

            if nutrient in nutrient_cols:
                model.addConstr(
                    quicksum(
                        foods_df.loc[foods_df["Food"] == food, nutrient].iloc[0] * var
                        for food, var in food_vars.items()
                    )
                    >= min_requirement,
                    name=f"{nutrient}_requirement",
                )
            else:
                logger.warning(
                    f"Nutrient '{nutrient}' not found in foods data. Skipping constraint."
                )

        logger.info("Solving optimization model...")

        # Solve the model
        model.optimize()

        # Enhanced status checking
        if model.status == GRB.OPTIMAL:
            logger.info("Optimal solution found successfully")
        elif model.status == GRB.INFEASIBLE:
            logger.error("Problem is infeasible")
            # Try to compute IIS (Irreducible Inconsistent Subsystem) for debugging
            try:
                model.computeIIS()
                iis_constraints = []
                for constr in model.getConstrs():
                    if constr.IISConstr:
                        iis_constraints.append(constr.ConstrName)
                if iis_constraints:
                    raise ValueError(
                        f"Problem is infeasible. Conflicting constraints: {iis_constraints}. Try reducing requirements or adding more diverse foods."
                    )
                else:
                    raise ValueError(
                        "Problem is infeasible. The nutritional requirements cannot be met with the provided foods. Try reducing requirements or adding more foods with different nutritional profiles."
                    )
            except Exception as iis_error:
                logger.warning(f"Could not compute IIS: {str(iis_error)}")
                raise ValueError(
                    "Problem is infeasible. The nutritional requirements cannot be met with the provided foods. Try reducing requirements or adding more diverse foods."
                )
        elif model.status == GRB.UNBOUNDED:
            logger.error("Problem is unbounded")
            raise ValueError(
                "Problem is unbounded. This usually occurs when some foods have zero cost. Please ensure all foods have positive costs."
            )
        elif model.status == GRB.TIME_LIMIT:
            logger.error("Time limit reached")
            raise ValueError(
                "Solver time limit reached (60 seconds). The problem may be too complex. Try simplifying the problem or contact support."
            )
        elif model.status == GRB.INTERRUPTED:
            logger.error("Solver was interrupted")
            raise ValueError("Solver was interrupted. Please try again.")
        elif model.status == GRB.NUMERIC:
            logger.error("Numerical difficulties encountered")
            raise ValueError(
                "Numerical difficulties encountered. Try using simpler numbers or scaling your data."
            )
        else:
            logger.error(f"Unexpected solver status: {model.status}")
            raise ValueError(
                f"Failed to find an optimal solution. Solver status: {model.status}. Please check your input data."
            )

    except Exception as gurobi_error:
        logger.error(f"Gurobi optimization error: {str(gurobi_error)}")
        if "Gurobi" in str(type(gurobi_error)):
            raise ValueError(
                f"Gurobi solver error: {str(gurobi_error)}. Please check your Gurobi installation and license."
            )
        else:
            raise

    # Extract results
    try:
        total_cost = model.objVal
        logger.info(f"Optimal solution found with total cost: {total_cost:.2f}")

        # Create results dataframe
        result_rows = []
        for food_name, var in food_vars.items():
            try:
                food_row = foods_df[foods_df["Food"] == food_name].iloc[0]
                amount = var.X

                result_row = {
                    "Food": food_name,
                    "Units": amount,
                    "Cost": food_row["Cost"] * amount,
                }

                # Add nutrient contributions
                for nutrient in nutrient_cols:
                    result_row[nutrient] = food_row[nutrient] * amount

                result_rows.append(result_row)
            except Exception as extract_error:
                logger.error(
                    f"Error extracting results for food '{food_name}': {str(extract_error)}"
                )
                raise ValueError(
                    f"Error processing results for food '{food_name}': {str(extract_error)}"
                )

        result_df = pd.DataFrame(result_rows)

        # Validate results
        if result_df.empty:
            raise ValueError(
                "No results generated. This is unexpected after finding an optimal solution."
            )

        # Check if any solution violates non-negativity (shouldn't happen, but good to check)
        if (result_df["Units"] < -1e-6).any():
            negative_foods = result_df[result_df["Units"] < -1e-6]["Food"].tolist()
            logger.warning(
                f"Negative amounts found for foods (numerical error): {negative_foods}"
            )
            # Clamp negative values to zero
            result_df["Units"] = result_df["Units"].clip(lower=0)

    except Exception as result_error:
        logger.error(f"Error extracting optimization results: {str(result_error)}")
        raise ValueError(
            f"Failed to extract optimization results: {str(result_error)}"
        )  # Create flexible visualizations
    try:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Diet Problem Optimization Results", fontsize=16, y=1.02)

        # Plot 1: Food Units - only show foods with positive amounts
        foods_with_amounts = result_df[result_df["Units"] > 0.001]
        if not foods_with_amounts.empty:
            colors = plt.cm.Set3(range(len(foods_with_amounts)))
            axs[0, 0].bar(
                foods_with_amounts["Food"], foods_with_amounts["Units"], color=colors
            )
            axs[0, 0].set_title("Optimal Food Quantities")
            axs[0, 0].set_ylabel("Units")
            axs[0, 0].tick_params(axis="x", rotation=45)
        else:
            axs[0, 0].text(
                0.5,
                0.5,
                "No foods selected\n(all amounts are zero)",
                ha="center",
                va="center",
                transform=axs[0, 0].transAxes,
            )
            axs[0, 0].set_title("Optimal Food Quantities")

        # Plot 2: Cost Breakdown
        if not foods_with_amounts.empty:
            axs[0, 1].bar(
                foods_with_amounts["Food"], foods_with_amounts["Cost"], color=colors
            )
            axs[0, 1].set_title("Cost per Food Type")
            axs[0, 1].set_ylabel("Cost ($)")
            axs[0, 1].tick_params(axis="x", rotation=45)
        else:
            axs[0, 1].text(
                0.5,
                0.5,
                "No costs to display",
                ha="center",
                va="center",
                transform=axs[0, 1].transAxes,
            )
            axs[0, 1].set_title("Cost per Food Type")

        # Plot 3: Nutritional Requirements vs Achieved
        achieved_nutrients = {}
        for nutrient in nutrient_cols:
            achieved_nutrients[nutrient] = result_df[nutrient].sum()

        requirements_dict = dict(
            zip(requirements_df["Nutrient"], requirements_df["Minimum"])
        )

        nutrients = list(nutrient_cols)
        requirements = [requirements_dict.get(nut, 0) for nut in nutrients]
        achieved = [achieved_nutrients.get(nut, 0) for nut in nutrients]

        if nutrients:
            x_pos = range(len(nutrients))
            width = 0.35
            axs[1, 0].bar(
                [i - width / 2 for i in x_pos],
                requirements,
                width,
                label="Required",
                color="red",
                alpha=0.7,
            )
            axs[1, 0].bar(
                [i + width / 2 for i in x_pos],
                achieved,
                width,
                label="Achieved",
                color="green",
                alpha=0.7,
            )
            axs[1, 0].set_title("Nutritional Requirements vs Achieved")
            axs[1, 0].set_ylabel("Units")
            axs[1, 0].set_xticks(x_pos)
            axs[1, 0].set_xticklabels(nutrients, rotation=45)
            axs[1, 0].legend()
        else:
            axs[1, 0].text(
                0.5,
                0.5,
                "No nutrients to display",
                ha="center",
                va="center",
                transform=axs[1, 0].transAxes,
            )
            axs[1, 0].set_title("Nutritional Requirements vs Achieved")

        # Plot 4: Summary Information
        summary_data = [("Total Cost", total_cost)]
        for nutrient in nutrient_cols:
            summary_data.append(
                (f"Total {nutrient}", achieved_nutrients.get(nutrient, 0))
            )

        if summary_data:
            summary_labels, summary_values = zip(*summary_data)
            colors_summary = plt.cm.viridis(range(len(summary_data)))

            bars = axs[1, 1].bar(summary_labels, summary_values, color=colors_summary)
            axs[1, 1].set_title("Diet Summary")
            axs[1, 1].set_ylabel("Value")

            # Add value labels on bars
            for bar, value in zip(bars, summary_values):
                height = bar.get_height()
                axs[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(summary_values) * 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axs[1, 1].tick_params(axis="x", rotation=45)
        else:
            axs[1, 1].text(
                0.5,
                0.5,
                "No summary data to display",
                ha="center",
                va="center",
                transform=axs[1, 1].transAxes,
            )
            axs[1, 1].set_title("Diet Summary")

        fig.tight_layout()

        logger.info("Visualization created successfully")

    except Exception as plot_error:
        logger.error(f"Error creating visualization: {str(plot_error)}")
        # Create a simple fallback plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"Visualization Error\nOptimal Cost: ${total_cost:.2f}\nSee results table for details",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Diet Problem Results")
        ax.axis("off")

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
