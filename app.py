import os

import gradio as gr
import pandas as pd

from models.gurobi_models import solve_diet_problem, solve_plne
from ui.gradio_sections import diet_problem_tab, project_info_tab, vehicle_routing_tab

# Mock Data for Vehicle Routing Problem
plne_df = pd.DataFrame(
    [
        {"Node": 0, "X": 50, "Y": 50, "Demand": 0},
        {"Node": 1, "X": 20, "Y": 20, "Demand": 10},
        {"Node": 2, "X": 80, "Y": 20, "Demand": 15},
        {"Node": 3, "X": 20, "Y": 80, "Demand": 10},
        {"Node": 4, "X": 80, "Y": 80, "Demand": 10},
        {"Node": 5, "X": 50, "Y": 10, "Demand": 20},
    ]
)

# Descriptions
plne_description = """
### 🚚 Capacitated Vehicle Routing Problem
Provide node coordinates and demands, plus vehicle capacity and number of vehicles.         
"""

diet_description = """
### 🍎 Diet Problem

Find the optimal diet that minimizes cost while meeting nutritional requirements.
"""

# Read and encode the PDF - go up one directory to find assets at project root
favicon_path = os.path.join(os.path.dirname(__file__), "assets", "favicon.ico")

# Assemble UI
with gr.Blocks(title="Operations Research App") as ro_app:
    gr.Markdown(
        """
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="https://insat.rnu.tn/assets/images/logo_c.png" width="100">
        <div>
            <h1 style="margin-bottom: 5px;">Operations Research Project</h1>
            <p style="margin-top: 0; font-size: 14px; color: gray;">Choose the convenient tab to solve a problem</p>
        </div>
    </div>
    """
    )
    with gr.Tabs():
        project_info_tab()
        diet_problem_tab(solve_diet_problem, diet_description)
        vehicle_routing_tab(plne_df, solve_plne, plne_description)

if __name__ == "__main__":
    ro_app.launch(favicon_path=favicon_path)
