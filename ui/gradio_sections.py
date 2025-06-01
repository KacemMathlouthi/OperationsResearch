import base64
import os

import gradio as gr
import pandas as pd

import utils.logger as logger

logger = logger.get_logger(__name__)


def project_info_tab():
    with gr.Tab("\U0001f4d8 Project Info"):
        gr.Markdown(
            """
            # \U0001f393 GL3 - 2025 - Operational Research Project
            This application demonstrates how **Linear Programming (PL)** and **Mixed-Integer Linear Programming (PLNE)** can be applied to solve real-world optimisation problems using **Gurobi**.
            """
        )

        gr.HTML(
            """
            <style>
                .member-card {
                    display: inline-block;
                    width: 160px;
                    text-align: center;
                    margin: 10px;
                }
                .member-card img {
                    width: 120px;
                    height: 140px;
                    object-fit: cover;
                    border-radius: 8px;
                    border: 1px solid #ccc;
                }
                .member-name {
                    margin-top: 6px;
                    font-weight: bold;
                }
            </style>

            <div style="display: flex; flex-wrap: wrap; justify-content: center;">
                <div class="member-card">
                    <img src="https://i.imgur.com/ff50RGn.jpeg" alt="Kacem Mathlouthi">
                    <div class="member-name">Kacem Mathlouthi</div>
                </div>
                <div class="member-card">
                    <img src="URL_HERE/mohamed_amine_haouas.jpg" alt="Mohamed Amine Houas">
                    <div class="member-name">Mohamed Amine Houas</div>
                </div>
                <div class="member-card">
                    <img src="URL_HERE/oussema_kraiem.jpg" alt="Oussema Kraiem">
                    <div class="member-name">Oussema Kraiem</div>
                </div>
                <div class="member-card">
                    <img src="URL_HERE/yassine_taieb.jpg" alt="Yassine Taieb">
                    <div class="member-name">Yassine Taieb</div>
                </div>
                <div class="member-card">
                    <img src="URL_HERE/youssef_sghairi.jpg" alt="Youssef Sghairi">
                    <div class="member-name">Youssef Sghairi</div>
                </div>
                <div class="member-card">
                    <img src="URL_HERE/youssef_aridhi.jpg" alt="Youssef Aaridhi">
                    <div class="member-name">Youssef Aaridhi</div>
                </div>
            </div>
            """
        )

        gr.Markdown(
            """
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


def diet_problem_tab(solve_diet_problem, diet_description):
    with gr.Tab("🍎 Diet Problem (PL)"):
        gr.Markdown(diet_description)

        # Add mathematical model description
        gr.Markdown(
            r"""
            ### 🧮 Mathematical Formulation

            **Parameters**
            | Symbol      | Description                                    |
            |-------------|------------------------------------------------|
            | $$I$$       | Set of available foods                         |
            | $$J$$       | Set of nutrients                               |
            | $$c_i$$     | Cost per unit of food i                        |
            | $$n_{ij}$$  | Amount of nutrient j in one unit of food i     |
            | $$R_j$$     | Minimum requirement for nutrient j             |

            **Decision Variables**
            | Symbol      | Description                             |
            |-------------|-----------------------------------------|
            | $$x_i$$     | Units of food i to consume     |

            **Objective Function:**  
            $$
            \text{Minimize} \quad Z = \sum_{i \in I} c_i \cdot x_i
            $$

            **Constraints:**  
            1. **Nutritional requirements:**
            $$\sum_{i \in I} n_{ij} \cdot x_i \geq R_j \quad \forall j \in J$$

            2. **Non-negativity:**
            $$x_i \geq 0 \quad \forall i \in I$$
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🍎 Foods Data")
                gr.Markdown(
                    "Add foods with their costs and nutritional content per unit:"
                )

                # Default foods data
                default_foods = pd.DataFrame(
                    [
                        {"Food": "Food A", "Cost": 3.0, "Protein": 2.0, "Fat": 1.0},
                        {"Food": "Food B", "Cost": 2.0, "Protein": 1.0, "Fat": 2.0},
                    ]
                )

                foods_input = gr.Dataframe(
                    value=default_foods,
                    headers=["Food", "Cost", "Protein", "Fat"],
                    datatype=["str", "number", "number", "number"],
                    col_count=(4, "dynamic"),
                    row_count=(2, "dynamic"),
                    label="Foods and Nutritional Content",
                    interactive=True,
                )

            with gr.Column():
                gr.Markdown("### 🥗 Nutritional Requirements")
                gr.Markdown("Specify minimum daily requirements for each nutrient:")

                # Default requirements data
                default_requirements = pd.DataFrame(
                    [
                        {"Nutrient": "Protein", "Minimum": 8.0},
                        {"Nutrient": "Fat", "Minimum": 6.0},
                    ]
                )

                requirements_input = gr.Dataframe(
                    value=default_requirements,
                    headers=["Nutrient", "Minimum"],
                    datatype=["str", "number"],
                    col_count=(2, "fixed"),
                    row_count=(2, "dynamic"),
                    label="Nutritional Requirements",
                    interactive=True,
                )

        solve_btn = gr.Button("Solve Diet Problem", variant="primary")
        status_output = gr.Textbox(label="Status", interactive=False)
        results_table = gr.Dataframe(label="Optimization Results")
        results_plot = gr.Plot(label="Results Visualization")

        def _solve_diet_optimization(foods_df, requirements_df):
            try:
                logger.info("Starting diet optimization from UI")

                # Input existence validation
                if foods_df is None or len(foods_df) == 0:
                    return (
                        pd.DataFrame(),
                        None,
                        "❌ Error: Please provide foods data. Add at least one food item with its cost and nutritional content.",
                    )

                if requirements_df is None or len(requirements_df) == 0:
                    return (
                        pd.DataFrame(),
                        None,
                        "❌ Error: Please provide requirements data. Add at least one nutritional requirement.",
                    )

                # Convert to DataFrame if needed
                if not isinstance(foods_df, pd.DataFrame):
                    try:
                        foods_df = pd.DataFrame(foods_df)
                    except Exception as e:
                        return (
                            pd.DataFrame(),
                            None,
                            f"❌ Error: Cannot convert foods data to DataFrame: {str(e)}",
                        )

                if not isinstance(requirements_df, pd.DataFrame):
                    try:
                        requirements_df = pd.DataFrame(requirements_df)
                    except Exception as e:
                        return (
                            pd.DataFrame(),
                            None,
                            f"❌ Error: Cannot convert requirements data to DataFrame: {str(e)}",
                        )

                # Remove completely empty rows
                foods_df = foods_df.dropna(how="all")
                requirements_df = requirements_df.dropna(how="all")

                # Check if data still exists after cleaning
                if foods_df.empty:
                    return (
                        pd.DataFrame(),
                        None,
                        "❌ Error: No valid foods data found. Please ensure at least one row has valid data.",
                    )

                if requirements_df.empty:
                    return (
                        pd.DataFrame(),
                        None,
                        "❌ Error: No valid requirements data found. Please ensure at least one row has valid data.",
                    )

                # Validate required columns
                if "Food" not in foods_df.columns or "Cost" not in foods_df.columns:
                    return (
                        pd.DataFrame(),
                        None,
                        "❌ Error: Foods data must have 'Food' and 'Cost' columns. Please check your column headers.",
                    )

                if (
                    "Nutrient" not in requirements_df.columns
                    or "Minimum" not in requirements_df.columns
                ):
                    return (
                        pd.DataFrame(),
                        None,
                        "❌ Error: Requirements data must have 'Nutrient' and 'Minimum' columns. Please check your column headers.",
                    )

                # Check for missing values in critical columns
                missing_food_names = foods_df["Food"].isna().sum()
                if missing_food_names > 0:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: {missing_food_names} food(s) have missing names. All foods must have valid names.",
                    )

                missing_costs = foods_df["Cost"].isna().sum()
                if missing_costs > 0:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: {missing_costs} food(s) have missing costs. All foods must have valid costs.",
                    )

                missing_nutrients = requirements_df["Nutrient"].isna().sum()
                if missing_nutrients > 0:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: {missing_nutrients} requirement(s) have missing nutrient names. All requirements must have valid nutrient names.",
                    )

                missing_minimums = requirements_df["Minimum"].isna().sum()
                if missing_minimums > 0:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: {missing_minimums} requirement(s) have missing minimum values. All requirements must have valid minimum values.",
                    )

                # Check for negative values
                try:
                    numeric_cols = foods_df.select_dtypes(include=[float, int]).columns
                    numeric_cols = [
                        col for col in numeric_cols if col != "Food"
                    ]  # Exclude non-numeric columns

                    if (
                        len(numeric_cols) > 0
                        and (foods_df[numeric_cols] < 0).any().any()
                    ):
                        negative_foods = []
                        for col in numeric_cols:
                            if (foods_df[col] < 0).any():
                                bad_foods = foods_df[foods_df[col] < 0]["Food"].tolist()
                                negative_foods.extend(
                                    [f"{food} ({col})" for food in bad_foods]
                                )

                        return (
                            pd.DataFrame(),
                            None,
                            f"❌ Error: Negative values found: {', '.join(negative_foods[:5])}{'...' if len(negative_foods) > 5 else ''}. All numeric values must be non-negative.",
                        )
                except Exception as numeric_error:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: Problem checking numeric values: {str(numeric_error)}. Please ensure all numeric columns contain valid numbers.",
                    )

                try:
                    if (requirements_df["Minimum"] <= 0).any():
                        bad_requirements = requirements_df[
                            requirements_df["Minimum"] <= 0
                        ]["Nutrient"].tolist()
                        return (
                            pd.DataFrame(),
                            None,
                            f"❌ Error: Non-positive requirements found for: {', '.join(bad_requirements)}. All requirements must be positive values.",
                        )
                except Exception as req_error:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: Problem checking requirements: {str(req_error)}. Please ensure all requirement values are positive numbers.",
                    )

                # Check for empty strings in food names
                empty_food_names = foods_df[
                    foods_df["Food"].astype(str).str.strip() == ""
                ]["Food"].count()
                if empty_food_names > 0:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: {empty_food_names} food(s) have empty names. All foods must have non-empty names.",
                    )

                # Check for empty strings in nutrient names
                empty_nutrient_names = requirements_df[
                    requirements_df["Nutrient"].astype(str).str.strip() == ""
                ]["Nutrient"].count()
                if empty_nutrient_names > 0:
                    return (
                        pd.DataFrame(),
                        None,
                        f"❌ Error: {empty_nutrient_names} nutrient(s) have empty names. All nutrients must have non-empty names.",
                    )

                logger.info("UI validation passed, calling solver...")
                result_df, fig = solve_diet_problem(foods_df, requirements_df)

                # Validate solver results
                if result_df is None or result_df.empty:
                    return (
                        pd.DataFrame(),
                        None,
                        "❌ Error: Solver returned empty results. This is unexpected.",
                    )

                # Check if solution makes sense
                total_cost = (
                    result_df["Cost"].sum() if "Cost" in result_df.columns else 0
                )
                if total_cost < 0:
                    logger.warning(f"Negative total cost detected: {total_cost}")

                logger.info(
                    f"Optimization completed successfully with total cost: {total_cost:.2f}"
                )
                return (
                    result_df,
                    fig,
                    f"✅ Solved Successfully! Optimal diet plan found with total cost: ${total_cost:.2f}",
                )

            except ValueError as ve:
                logger.error(f"Validation error: {str(ve)}")
                return pd.DataFrame(), None, f"❌ Validation Error: {str(ve)}"
            except TypeError as te:
                logger.error(f"Type error: {str(te)}")
                return pd.DataFrame(), None, f"❌ Data Type Error: {str(te)}"
            except Exception as e:
                logger.error(f"Unexpected error in diet optimization: {str(e)}")
                error_msg = str(e)
                if "Gurobi" in error_msg:
                    return pd.DataFrame(), None, f"❌ Solver Error: {error_msg}"
                elif "infeasible" in error_msg.lower():
                    return pd.DataFrame(), None, f"❌ Infeasible Problem: {error_msg}"
                elif "unbounded" in error_msg.lower():
                    return pd.DataFrame(), None, f"❌ Unbounded Problem: {error_msg}"
                else:
                    return pd.DataFrame(), None, f"❌ Unexpected Error: {error_msg}"

        solve_btn.click(
            fn=_solve_diet_optimization,
            inputs=[foods_input, requirements_input],
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
