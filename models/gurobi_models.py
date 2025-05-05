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


def mock_solve_plne(data):
    df = pd.DataFrame(
        {"Employee": data["Employee"], "Assigned Shift": ["Shift 1", "Shift 2", "Off"]}
    )
    fig, ax = plt.subplots()
    ax.pie([1, 1, 1], labels=["Shift 1", "Shift 2", "Off"], autopct="%1.1f%%")
    ax.set_title("Mock Shift Allocation")
    return df, fig
