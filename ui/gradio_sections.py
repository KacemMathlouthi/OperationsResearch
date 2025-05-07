import gradio as gr
import os
import base64
import sys
import pandas as pd                                                        
import achref.src.logger as logger

logger = logger.get_logger(__name__)

def project_info_tab():
    with gr.Tab("📘 Project Info"):
        gr.Markdown(
            """
        # 🎓 GL3 - 2025 - Operational Research Project
        This application demonstrates how **Linear Programming (PL)** and **Mixed-Integer Linear Programming (PLNE)** can be applied to solve real-world optimisation problems using **Gurobi**.
        
        ---
        # 👥 Project Members
        - **Kacem Mathlouthi** — GL3/2  
        - **Mohamed Amine Houas** — GL3/1  
        - **Oussema Kraiem** — GL3/2  
        - **Yassine Taieb** — GL3/2  
        - **Youssef Sghairi** — GL3/2  
        - **Youssef Aaridhi** — GL3/2  
        - **Achref Ben Ammar** — GL3/1  
        ---
        # 🧾 Compte Rendu
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


def production_planning_tab(mock_pl_df, solve_pl_gurobi, pl_description):
    with gr.Tab("🏭 Production Planning (PL)"):
        gr.Markdown(pl_description)

        # Add mathematical model description
        gr.Markdown(
            r"""
            ### 🧮 Mathematical Formulation

            Let:  
            | Symbol      | Description                             |
            |-------------|-----------------------------------------|
            | $$x_i$$       | Number of units to produce for product i |
            | $$p_i$$       | Profit per unit for product i         |
            | $$r_i$$       | Resource usage per unit for product i |
            | $$R$$         | Total resource available              |

            **Objective:**  
            $$
            \text{Maximise} \quad \sum_i p_i \cdot x_i
            $$

            **Constraint:**  
            $$
            \sum_i r_i \cdot x_i \leq R \quad \text{and} \quad x_i \geq 0
            $$
            """
        )

        with gr.Row():
            input_pl = gr.Dataframe(
                headers=["Product", "Profit/Unit", "Resource Usage"],
                value=mock_pl_df,
                label="Input Product Data",
            )
        total_resource_input = gr.Number(
            value=100, label="Total Resource Available (R)"
        )
        solve_btn_pl = gr.Button("Solve Production Problem")

        result_table_pl = gr.Dataframe(label="Optimised Result")
        
        # Create plot output placeholders
        result_plot_combined = gr.Plot(label="Data Visualisation")

        def _solve_with_floats(df, R):
                df["Profit/Unit"]     = df["Profit/Unit"].astype(float)
                df["Resource Usage"]  = df["Resource Usage"].astype(float)
                return solve_pl_gurobi(df, total_resource=R)
        
        solve_btn_pl.click(
            fn=_solve_with_floats,
            inputs=[input_pl, total_resource_input],
            outputs=[
                result_table_pl,
                result_plot_combined,
            ]
        )


# in gradio_sections.py

def vehicle_routing_tab(mock_plne_df, solve_plne, plne_description):
    
    with gr.Tab("🚚 Vehicle Routing (PLNE)"):
        gr.Markdown(plne_description)
        # Log the Python path of the project
        gr.HTML(
            '<img src="https://pyvrp.readthedocs.io/en/latest/_images/introduction-to-vrp.svg" '
            'alt="VRP Problem Illustration" width="600px" />'
        )
        gr.Markdown(
            r"""
            ### 🧮 Mathematical Formulation (Capacitated VRP)


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
            > *Explanation:* Every customer `i` must have exactly one vehicle leaving it and one arriving—ensuring each customer is visited exactly once.

            2. **Depot flow**  
            $$
            \sum_{j>0} x_{0j} = K
            $$
            $$
            \sum_{i>0} x_{i0} = K
            $$
            > *Explanation:* Exactly `K` vehicles depart from the depot and `K` return, so all vehicles are used and end back at the depot.

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
            > *Explanation:*  
            > - If `x_{ij}=1`, then $$u_j \ge u_i + d_j$$ enforcing vehicle capacity.  
            > - These constraints also prevent any customer‐only loops (subtours), because load can’t reset without returning to the depot.  
            > - We fix `u_0=0` at the depot and bound `u_i` by capacity `Q`.


            ---

           

"""
        )
    
        vrp_input = gr.Dataframe(
                    headers=["Node", "X", "Y", "Demand"],
                    value=mock_plne_df,
                    label="Input Vehicle Routing Data",
                )
        with gr.Row():
            cap_input = gr.Number(value=40, label="Vehicle capacity (Q)")
            k_input   = gr.Number(value=2, label="Number of vehicles (K)")
        solve_btn = gr.Button("Solve VRP")
        status_output = gr.Textbox(label="Status", interactive=False)
        result_table = gr.Dataframe(label="Routes Summary")
        result_plot  = gr.Plot(label="Route Map & Summary")

        def _solve_vrp_with_floats(df, Q, K):
            df["X"]      = df["X"].astype(float)
            df["Y"]      = df["Y"].astype(float)
            df["Demand"] = df["Demand"].astype(float)
            
            # skip depot (assumed Node==0) when checking
            custs = df[df["Node"] != 0]

            try:
                # 1) any single demand > Q?
                too_big = custs[custs["Demand"] > Q]
                if not too_big.empty:
                    bad = int(too_big["Node"].iloc[0])
                    raise ValueError(f"Client {bad} demand ({too_big['Demand'].iloc[0]}) exceeds capacity Q={Q}")

                # 2) total demand > Q*K?
                total = custs["Demand"].sum()
                if total > Q * K:
                    raise ValueError(f"Total demand ({total}) exceeds fleet capacity Q*K={Q*K}")

                # all good → call solver
                routes_df, fig = solve_plne(df, vehicle_capacity=Q, num_vehicles=K)
                return routes_df, fig, "All Good"
            except ValueError as e:
                # on error show empty table/plot + message
                return pd.DataFrame(), None, str(e)
        
        
        solve_btn.click(
            fn=_solve_vrp_with_floats,
            inputs=[vrp_input, cap_input, k_input],
            outputs=[result_table, result_plot, status_output],
        )

