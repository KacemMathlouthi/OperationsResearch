import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Mock function for solving
def mock_solve_pl(data):
    quantity = [10] * len(data)
    profit_per_unit = data["Profit/Unit"]
    total_profit = [q * p for q, p in zip(quantity, profit_per_unit)]

    df = pd.DataFrame({
        "Product": data["Product"],
        "Quantity Produced": quantity,
        "Profit": total_profit
    })

    fig, ax = plt.subplots()
    ax.bar(df["Product"], df["Quantity Produced"])
    ax.set_title("Mock Production Output")

    return df, fig


def mock_solve_plne(data):
    df = pd.DataFrame({
        "Employee": data["Employee"],
        "Assigned Shift": ["Shift 1", "Shift 2", "Off"]
    })
    fig, ax = plt.subplots()
    ax.pie([1, 1, 1], labels=["Shift 1", "Shift 2", "Off"], autopct="%1.1f%%")
    ax.set_title("Mock Shift Allocation")
    return df, fig

# --- Mock data
mock_pl_df = pd.DataFrame({
    "Product": ["A", "B", "C"],
    "Profit/Unit": [10, 20, 15],
    "Resource Usage": [3, 5, 2]
})

mock_plne_df = pd.DataFrame({
    "Employee": ["Alice", "Bob", "Charlie"],
    "Availability": ["Yes", "Yes", "No"]
})

# --- Problem Descriptions
pl_description = """
### 🏭 Production Planning (PL)
Select the quantity of each product to produce to **maximize profit**, under limited resource constraints.
"""

plne_description = """
### 👥 Staff Scheduling (PLNE)
Assign employees to shifts to **minimize cost** or **maximize shift coverage**, with availability and legal limits.
"""

# --- Gradio Tabs
with gr.Blocks() as demo:
    gr.Markdown("# 🔧 Operations Research Project")

    with gr.Tabs():
        with gr.Tab("Production Planning (PL)"):
            gr.Markdown(pl_description)
            with gr.Row():
                input_pl = gr.Dataframe(
                    headers=["Product", "Profit/Unit", "Resource Usage"],
                    value=mock_pl_df,
                    label="Input Product Data"
                )
            solve_btn_pl = gr.Button("Solve Production Problem")
            result_table_pl = gr.Dataframe(label="Optimised Result (Mock)")
            result_plot_pl = gr.Plot(label="Visualisation")

            solve_btn_pl.click(fn=mock_solve_pl, inputs=input_pl, outputs=[result_table_pl, result_plot_pl])

        with gr.Tab("Staff Scheduling (PLNE)"):
            gr.Markdown(plne_description)
            with gr.Row():
                input_plne = gr.Dataframe(
                    headers=["Employee", "Availability"],
                    value=mock_plne_df,
                    label="Input Staff Availability"
                )
            solve_btn_plne = gr.Button("Solve Scheduling Problem")
            result_table_plne = gr.Dataframe(label="Assignment Result (Mock)")
            result_plot_plne = gr.Plot(label="Visualisation")

            solve_btn_plne.click(fn=mock_solve_plne, inputs=input_plne, outputs=[result_table_plne, result_plot_plne])

# --- Run App
if __name__ == "__main__":
    demo.launch()
