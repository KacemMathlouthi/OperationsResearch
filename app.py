import os

import gradio as gr
import pandas as pd

from models.gurobi_models import solve_plne, solve_refinery_optimization
from ui.gradio_sections import oil_refinery_tab, project_info_tab, vehicle_routing_tab

# Mock Data
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

# Oil Refinery Optimization Mock Data
crude_df = pd.DataFrame(
    [
        {"Crude": "Light Crude", "Cost": 60, "Availability": 10000},
        {"Crude": "Medium Crude", "Cost": 50, "Availability": 15000},
        {"Crude": "Heavy Crude", "Cost": 45, "Availability": 12000},
    ]
)

product_df = pd.DataFrame(
    [
        {"Product": "Premium Gasoline", "Price": 90, "Demand": 5000},
        {"Product": "Regular Gasoline", "Price": 80, "Demand": 7000},
        {"Product": "Diesel", "Price": 75, "Demand": 8000},
    ]
)

yields_df = pd.DataFrame(
    [
        # Light Crude yields
        {
            "Crude": "Light Crude",
            "Product": "Premium Gasoline",
            "Yield": 0.4,
            "Quality": 95,
        },
        {
            "Crude": "Light Crude",
            "Product": "Regular Gasoline",
            "Yield": 0.3,
            "Quality": 90,
        },
        {"Crude": "Light Crude", "Product": "Diesel", "Yield": 0.2, "Quality": 85},
        # Medium Crude yields
        {
            "Crude": "Medium Crude",
            "Product": "Premium Gasoline",
            "Yield": 0.3,
            "Quality": 85,
        },
        {
            "Crude": "Medium Crude",
            "Product": "Regular Gasoline",
            "Yield": 0.4,
            "Quality": 80,
        },
        {"Crude": "Medium Crude", "Product": "Diesel", "Yield": 0.3, "Quality": 80},
        # Heavy Crude yields
        {
            "Crude": "Heavy Crude",
            "Product": "Premium Gasoline",
            "Yield": 0.1,
            "Quality": 75,
        },
        {
            "Crude": "Heavy Crude",
            "Product": "Regular Gasoline",
            "Yield": 0.3,
            "Quality": 70,
        },
        {"Crude": "Heavy Crude", "Product": "Diesel", "Yield": 0.5, "Quality": 75},
    ]
)

quality_reqs_df = pd.DataFrame(
    [
        {"Product": "Premium Gasoline", "MinQuality": 90},
        {"Product": "Regular Gasoline", "MinQuality": 80},
        {"Product": "Diesel", "MinQuality": 75},
    ]
)

# Descriptions
plne_description = """
### 🚚 Capacitated Vehicle Routing Problem
Provide node coordinates and demands, plus vehicle capacity and number of vehicles.         
"""

refinery_description = """
### ⚙️ Oil Refinery Optimization Problem

**Scenario**
An oil refinery wants to determine the optimal production plan for different fuel products (like diesel, premium gasoline, and regular gasoline) 
using various crude oils. Each crude oil has different yields, costs, and qualities, and each product has its own demand, 
quality requirements, and selling price.
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
        oil_refinery_tab(
            crude_df,
            product_df,
            yields_df,
            quality_reqs_df,
            solve_refinery_optimization,
            refinery_description,
        )
        vehicle_routing_tab(plne_df, solve_plne, plne_description)

if __name__ == "__main__":
    ro_app.launch(favicon_path=favicon_path)
