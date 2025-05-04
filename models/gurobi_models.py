import pandas as pd
import matplotlib.pyplot as plt


# Mock Solver Functions
def mock_solve_pl(data):
    quantity = [10] * len(data)
    profit_per_unit = data["Profit/Unit"]
    total_profit = [q * p for q, p in zip(quantity, profit_per_unit)]
    df = pd.DataFrame(
        {
            "Product": data["Product"],
            "Quantity Produced": quantity,
            "Profit": total_profit,
        }
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(df["Product"], df["Quantity Produced"])
    ax.set_title("Mock Production Output")
    return df, fig


def mock_solve_plne(data):
    df = pd.DataFrame(
        {"Employee": data["Employee"], "Assigned Shift": ["Shift 1", "Shift 2", "Off"]}
    )
    fig, ax = plt.subplots()
    ax.pie([1, 1, 1], labels=["Shift 1", "Shift 2", "Off"], autopct="%1.1f%%")
    ax.set_title("Mock Shift Allocation")
    return df, fig
