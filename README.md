<h1 align="center">Operations Research Web Application</h1>

<p align="center">
  <img src="https://insat.rnu.tn/assets/images/logo_c.png" width="100" alt="INSAT Logo">
</p>

This project demonstrates the use of **Linear Programming (PL)** and **Mixed-Integer Linear Programming (PLNE)** for solving real-world optimisation problems using **Gurobi**. It uses **Gradio** to provide an interactive web interface.

## Features

- **Production Planning (PL):** Optimises the number of products to manufacture for maximum profit under resource constraints.
- **Staff Scheduling (PLNE):** Mock assignment of employees to shifts based on availability.

## Project Structure

```
.
├── app.py                          # Main entry point of the Gradio application
├── assets/
│   └── compte_rendu.pdf           # Project report
├── models/
│   └── gurobi_models.py           # Gurobi-based solvers for PL and PLNE
├── ui/
│   └── gradio_sections.py         # UI layout and Gradio component logic
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
````

## Prerequisites
- Python 3.9 or higher

## Environment Setup
### 1. Clone the repository

```bash
git clone https://github.com/KacemMathlouthi/OperationsResearch.git
cd OperationsResearch
````

### 2. Create and activate a virtual environment

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Ensure you are in the project root directory and your virtual environment is activated:

```bash
python app.py
```

The application will launch locally at `http://127.0.0.1:7860/`.

## Usage

### Tabs Available:

* **Project Info:** Displays team information and a PDF report.
* **Production Planning (PL):** Solve and visualise a linear programming problem using product and resource data.
* **Staff Scheduling (PLNE):** Simulated assignment of employees to shifts based on availability.

## Notes

* Visualisations are generated with `matplotlib`.
* UI built with `Gradio Blocks` using tabbed layout.
* PDF report embedded with base64 encoding.
