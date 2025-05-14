import base64
import os

import gradio as gr
import pandas as pd

import achref.src.logger as logger

logger = logger.get_logger(__name__)


def project_info_tab():
    with gr.Tab("\U0001f4d8 Project Info"):
        gr.Markdown(
            """
        # \U0001f393 GL3 - 2025 - Operational Research Project
        This application demonstrates how **Linear Programming (PL)** and **Mixed-Integer Linear Programming (PLNE)** can be applied to solve real-world optimisation problems using **Gurobi**.
        
        ---
        # \U0001f465 Project Members
        - **Kacem Mathlouthi** — GL3/2  
        - **Mohamed Amine Houas** — GL3/1  
        - **Oussema Kraiem** — GL3/2  
        - **Yassine Taieb** — GL3/2  
        - **Youssef Sghairi** — GL3/2  
        - **Youssef Aaridhi** — GL3/2  
        - **Achref Ben Ammar** — GL3/1  
        ---
        # \U0001f9fe Compte Rendu
        """
        )
        pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets", "compte_rendu.pdf"
        )
        with open(pdf_path, "rb") as pdf_file:
            encoded_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")

        # Display using data URI
        gr.HTML(
            f"""
        <embed src="data:application/pdf;base64,{encoded_pdf}" type="application/pdf" width="100%" height="1200px">
        """
        )


def oil_refinery_tab(
    mock_crude_data,
    mock_product_data,
    mock_yields_data,
    mock_quality_reqs,
    solve_refinery_optimization,
    refinery_description,
):
    with gr.Tab("\U0001f3ed Oil Refinery Optimization (PL)"):
        gr.Markdown(refinery_description)

        # Add mathematical model description
        gr.Markdown(
            r"""
            ### 🧮 Mathematical Formulation

            **Sets and Indices**
            | Symbol      | Description                             |
            |-------------|-----------------------------------------|
            | $$i=1,...,m$$       | Types of crude oil (inputs) |
            | $$j=1,...,n$$       | Types of fuel products (outputs) |

            **Parameters**
            | Symbol      | Description                             |
            |-------------|-----------------------------------------|
            | $$c_i$$       | Cost per unit of crude $i$ |
            | $$p_j$$       | Selling price per unit of product $j$ |
            | $$y_{ij}$$    | Yield of product $j$ from crude $i$ (liters of product per liter of crude) |
            | $$q_{ij}$$    | Quality contribution of crude $i$ to product $j$ |
            | $$Q_j^{min}$$ | Minimum average quality required for product $j$ |
            | $$D_j$$       | Minimum demand (liters) for product $j$ |
            | $$A_i$$       | Availability limit (liters) of crude $i$ |

            **Decision Variables**
            | Symbol      | Description                             |
            |-------------|-----------------------------------------|
            | $$x_i$$       | Amount of crude oil $i$ used (liters) |

            **Objective:**  
            $$
            \text{Maximize} \quad \left(\sum_{j=1}^n p_j \cdot \sum_{i=1}^m y_{ij} \cdot x_i - \sum_{i=1}^m c_i \cdot x_i\right)
            $$

            **Constraints:**  
            1. Crude Availability:
            $$x_i \leq A_i \quad \forall i$$

            2. Demand Satisfaction:
            $$\sum_{i=1}^m y_{ij} \cdot x_i \geq D_j \quad \forall j$$

            3. Quality Requirements:
            $$\sum_{i=1}^m q_{ij} \cdot y_{ij} \cdot x_i \geq Q_j^{min} \cdot \sum_{i=1}^m y_{ij} \cdot x_i \quad \forall j$$

            4. Non-negativity:
            $$x_i \geq 0 \quad \forall i$$
            """
        )

        with gr.Tabs():
            with gr.TabItem("Crude Oils"):
                crude_input = gr.Dataframe(
                    headers=["Crude", "Cost", "Availability"],
                    value=mock_crude_data,
                    label="Crude Oil Data",
                )

            with gr.TabItem("Products"):
                product_input = gr.Dataframe(
                    headers=["Product", "Price", "Demand"],
                    value=mock_product_data,
                    label="Product Data",
                )

            with gr.TabItem("Yields & Quality"):
                yields_input = gr.Dataframe(
                    headers=["Crude", "Product", "Yield", "Quality"],
                    value=mock_yields_data,
                    label="Yield & Quality Data",
                )

            with gr.TabItem("Quality Requirements"):
                quality_reqs_input = gr.Dataframe(
                    headers=["Product", "MinQuality"],
                    value=mock_quality_reqs,
                    label="Quality Requirements",
                )

        solve_btn = gr.Button("Solve Refinery Optimization Problem")
        status_output = gr.Textbox(label="Status", interactive=False)
        results_table = gr.Dataframe(label="Optimization Results")
        results_plot = gr.Plot(label="Results Visualization")

        def _solve_refinery_problem(crude_df, product_df, yields_df, quality_df):
            try:
                # Convert all numeric columns to float
                for df in [crude_df, product_df, yields_df, quality_df]:
                    for col in df.columns:
                        if col not in ["Crude", "Product"]:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                result_df, fig = solve_refinery_optimization(
                    crude_df, product_df, yields_df, quality_df
                )
                return result_df, fig, "Solved Successfully"
            except Exception as e:
                return pd.DataFrame(), None, f"❌ Error: {str(e)}"

        solve_btn.click(
            fn=_solve_refinery_problem,
            inputs=[crude_input, product_input, yields_input, quality_reqs_input],
            outputs=[results_table, results_plot, status_output],
        )


def vehicle_routing_tab(mock_plne_df, solve_plne, plne_description):
    with gr.Tab("\U0001f69a Vehicle Routing (PLNE)"):
        gr.Markdown(plne_description)
        gr.HTML(
            '<img src="https://pyvrp.readthedocs.io/en/latest/_images/introduction-to-vrp.svg" '
            'alt="VRP Problem Illustration" width="600px" />'
        )
        gr.Markdown(
            r"""
            ### \U0001F9EE Mathematical Formulation (Capacitated VRP)


            | Symbol                         | Description                                                   |
            |--------------------------------|---------------------------------------------------------------|
            | $$i,j \in N=\{0,\dots,n\}$$    | Nodes (0 = depot, 1..n = customers)                           |
            | $$K$$                          | Number of vehicles                                            |
            | $$c_{ij}$$                     | Travel cost (distance) from node `i` to node `j`              |  
            | $$d_i$$                        | Demand at customer `i`                                        |
            | $$Q$$                          | Vehicle capacity                                              |
            | $$x_{ij}\in\{0,1\}$$           | 1 if a vehicle travels directly from `i` to `j`               |
            | $$u_i\ge0$$                    | Load on the vehicle immediately after visiting node `i`       |

            **Objective**  
            $$
            \min \sum_{i\in N}\sum_{\substack{j\in N \\ j\neq i}} c_{ij}\,x_{ij}
            $$  
            Minimize the **total travel cost** of all vehicles.

            ---

            **Subject to**

            1. **Degree constraints**  
            $$
            \sum_{j\neq i} x_{ij} = 1
            \quad \forall\, i\neq0
            $$
            $$
            \sum_{i\neq j} x_{ij} = 1
            \quad \forall\, j\neq0
            $$

            2. **Depot flow**  
            $$
            \sum_{j>0} x_{0j} = K
            $$
            $$
            \sum_{i>0} x_{i0} = K
            $$

            3. **MTZ subtour-elimination & capacity**  
            $$
            u_i - u_j + Q\,x_{ij} \le Q - d_j
            \quad \forall\,i\neq j,\; i,j>0
            $$
            $$
            u_0 = 0
            $$
            $$
            0 \le u_i \le Q
            $$
            """
        )

        vrp_input = gr.Dataframe(
            headers=["Node", "X", "Y", "Demand"],
            value=mock_plne_df,
            label="Input Vehicle Routing Data",
        )
        with gr.Row():
            cap_input = gr.Number(value=40, label="Vehicle capacity (Q)")
            k_input = gr.Number(value=2, label="Number of vehicles (K)")
        solve_btn = gr.Button("Solve VRP")
        status_output = gr.Textbox(label="Status", interactive=False)
        result_table = gr.Dataframe(label="Routes Summary")
        result_plot = gr.Plot(label="Route Map & Summary")

        def _solve_vrp_with_floats(df, Q, K):
            try:
                df["X"] = df["X"].astype(float)
                df["Y"] = df["Y"].astype(float)
                df["Demand"] = df["Demand"].astype(float)

                custs = df[df["Node"] != 0]

                too_big = custs[custs["Demand"] > Q]
                if not too_big.empty:
                    bad = int(too_big["Node"].iloc[0])
                    raise ValueError(
                        f"Client {bad} demand ({too_big['Demand'].iloc[0]}) exceeds capacity Q={Q}"
                    )

                total = custs["Demand"].sum()
                if total > Q * K:
                    raise ValueError(
                        f"Total demand ({total}) exceeds fleet capacity Q*K={Q*K}"
                    )

                routes_df, fig = solve_plne(df, vehicle_capacity=Q, num_vehicles=K)
                return routes_df, fig, "Solved Successfully"
            except Exception as e:
                return pd.DataFrame(), None, f"❌ Error: {str(e)}"

        solve_btn.click(
            fn=_solve_vrp_with_floats,
            inputs=[vrp_input, cap_input, k_input],
            outputs=[result_table, result_plot, status_output],
        )
