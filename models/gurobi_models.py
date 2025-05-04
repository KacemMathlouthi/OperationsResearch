from gurobipy import Model, GRB
import pandas as pd
import matplotlib.pyplot as plt

def solve_pl(data, total_resource=100):
    model = Model("ProductionPlanning")
    
    # Turn off solver output
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

    # Plot result
    fig, ax = plt.subplots()
    ax.bar(result_df["Product"], result_df["Quantity Produced"])
    ax.set_title("Optimised Production Output")
    ax.set_ylabel("Units Produced")

    return result_df, fig



def mock_solve_plne(data):
    df = pd.DataFrame(
        {"Employee": data["Employee"], "Assigned Shift": ["Shift 1", "Shift 2", "Off"]}
    )
    fig, ax = plt.subplots()
    ax.pie([1, 1, 1], labels=["Shift 1", "Shift 2", "Off"], autopct="%1.1f%%")
    ax.set_title("Mock Shift Allocation")
    return df, fig
