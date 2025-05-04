import gradio as gr
import pandas as pd
import os
from ui.gradio_sections import (
    project_info_tab,
    production_planning_tab,
    staff_scheduling_tab
)
from models.gurobi_models import (
    mock_solve_pl,
    mock_solve_plne
)

# Mock Data
mock_pl_df = pd.DataFrame({
    "Product": ["A", "B", "C"],
    "Profit/Unit": [10, 20, 15],
    "Resource Usage": [3, 5, 2]
})

mock_plne_df = pd.DataFrame({
    "Employee": ["Alice", "Bob", "Charlie"],
    "Availability": ["Yes", "Yes", "No"]
})

# Descriptions
pl_description = """
### 🏭 Production Planning (PL)
Select the quantity of each product to produce to **maximize profit**, under limited resource constraints.
"""

plne_description = """
### 👥 Staff Scheduling (PLNE)
Assign employees to shifts to **minimize cost** or **maximize shift coverage**, with availability and legal limits.
"""

# Read and encode the PDF - go up one directory to find assets at project root
favicon_path = os.path.join(os.path.dirname(__file__), "assets", "favicon.ico")

# Assemble UI
with gr.Blocks(title="Operations Research App") as ro_app:
    gr.Markdown("""
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="https://insat.rnu.tn/assets/images/logo_c.png" width="100">
        <div>
            <h1 style="margin-bottom: 5px;">Operations Research Project</h1>
            <p style="margin-top: 0; font-size: 14px; color: gray;">Choose the convenient tab to solve a problem</p>
        </div>
    </div>
    """)
    with gr.Tabs():
        project_info_tab()
        production_planning_tab(mock_pl_df, mock_solve_pl, pl_description)
        staff_scheduling_tab(mock_plne_df, mock_solve_plne, plne_description)

if __name__ == "__main__":
    ro_app.launch(favicon_path=favicon_path)
