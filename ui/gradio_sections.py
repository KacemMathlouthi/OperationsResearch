import gradio as gr
import os
import base64


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


def production_planning_tab(mock_pl_df, mock_solve_pl, pl_description):
    with gr.Tab("🏭 Production Planning (PL)"):
        gr.Markdown(pl_description)
        with gr.Row():
            input_pl = gr.Dataframe(
                headers=["Product", "Profit/Unit", "Resource Usage"],
                value=mock_pl_df,
                label="Input Product Data",
            )
        solve_btn_pl = gr.Button("Solve Production Problem")
        result_table_pl = gr.Dataframe(label="Optimised Result (Mock)")
        result_plot_pl = gr.Plot(label="Visualisation")

        solve_btn_pl.click(
            fn=mock_solve_pl, inputs=input_pl, outputs=[result_table_pl, result_plot_pl]
        )


def staff_scheduling_tab(mock_plne_df, mock_solve_plne, plne_description):
    with gr.Tab("👥 Staff Scheduling (PLNE)"):
        gr.Markdown(plne_description)
        with gr.Row():
            input_plne = gr.Dataframe(
                headers=["Employee", "Availability"],
                value=mock_plne_df,
                label="Input Staff Availability",
            )
        solve_btn_plne = gr.Button("Solve Scheduling Problem")
        result_table_plne = gr.Dataframe(label="Assignment Result (Mock)")
        result_plot_plne = gr.Plot(label="Visualisation")

        solve_btn_plne.click(
            fn=mock_solve_plne,
            inputs=input_plne,
            outputs=[result_table_plne, result_plot_plne],
        )
