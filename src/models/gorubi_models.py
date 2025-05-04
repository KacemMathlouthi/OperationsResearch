import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import combinations

# ---------------------------
# Sample Data Initialization
# ---------------------------
satellites = ["Sat1", "Sat2"]
debris = ["Deb1"]
time_steps = list(range(3))  # T=0,1,2
dt = 10  # Time step duration (seconds)
F_max = 500  # Max thrust (N)
m_i = 1000  # Satellite mass (kg)
d_safe = 10000  # 10 km safety distance (meters)
c_fuel = 0.01  # Fuel cost (kg/N)
M = 1e6  # Big-M value

# Initial positions and velocities (3D: x,y,z)
initial_state = {
    "Sat1": {
        "position": [100000, 0, 0],
        "velocity": [0, 7660, 0]  # ~Low Earth Orbit velocity
    },
    "Sat2": {
        "position": [0, 100000, 0],
        "velocity": [0, 7660, 0]
    },
    "Deb1": {
        "position": [50000, 50000, 0],
        "velocity": [0, 7660, 0]
    }
}

initial_position = {
    "Sat1": initial_state["Sat1"]["position"],  # x, y, z in meters
    "Sat2": initial_state["Sat2"]["position"],  # x, y, z in meters
    "Deb1": initial_state["Deb1"]["position"]  # x, y, z in meters
}
# ---------------------------
# Common Helper Functions
# ---------------------------
def add_orbital_dynamics(model, sat, time_steps, dt, mass, u, initial_position):
    # Initialize position and velocity variables
    p = {t: model.addVars(3, name=f"p_{sat}_{t}") for t in time_steps}
    v = {t: model.addVars(3, name=f"v_{sat}_{t}") for t in time_steps}
    
    # Set initial positions
    for coord in range(3):
        model.addConstr(p[0][coord] == initial_position[sat][coord])
    
    # Dynamics constraints
    for t in time_steps[:-1]:
        for coord in range(3):
            # Position update: p_{t+1} = p_t + v_t * dt + 0.5 * (u_t/mass) * dt^2
            model.addConstr(
                p[t+1][coord] == p[t][coord] + v[t][coord] * dt + 0.5 * (u[sat, t][coord]/mass) * dt**2
            )
            # Velocity update: v_{t+1} = v_t + (u_t/mass) * dt
            model.addConstr(
                v[t+1][coord] == v[t][coord] + (u[sat, t][coord]/mass) * dt
            )
    return p, v

# ---------------------------
# LP Model (Continuous Thrust)
# ---------------------------
def solve_lp_model():
    model = gp.Model("SatelliteLP")
    
    # Decision Variables
    u = {}  # Thrust vectors
    delta = {}  # Fuel consumption
    
    for sat in satellites:
        for t in time_steps[:-1]:  # No thrust at final step
            # Thrust vector components (-F_max to F_max)
            u[sat, t] = model.addVars(3, lb=-F_max, ub=F_max, name=f"u_{sat}_{t}")

            
            # Fuel consumption (linearized L1 norm)
            delta[sat, t] = model.addVar(
                name=f"delta_{sat}_{t}"
            )
            # Linearize absolute value: delta = sum(|u_xyz|) * c_fuel
            for coord in range(3):
                abs_u = model.addVar(name=f"abs_u_{sat}_{t}_{coord}")
                model.addGenConstrAbs(abs_u, u[sat, t][coord])
                model.addConstr(
                    delta[sat, t] >= c_fuel * abs_u
                )
    
    # Add orbital dynamics and position variables
    positions = {}
    velocities = {}

    for sat in satellites:
        p, v = add_orbital_dynamics(model, sat, time_steps, dt, m_i, u, initial_position)
        positions[sat] = p
        velocities[sat] = v
    
    # Collision Avoidance Constraints
    for t in time_steps:
        # Satellite-to-satellite
        for (i, j) in combinations(satellites, 2):
            for coord in range(3):
                sep = model.addVar(lb=d_safe/3, name=f"sep_{i}_{j}_{t}_{coord}")
                model.addConstr(
                    sep == positions[i][t][coord] - positions[j][t][coord]
                )
        
        # Satellite-to-debris
        for sat in satellites:
            for coord in range(3):
                sep = model.addVar(lb=d_safe/3, name=f"sep_{sat}_Deb1_{t}_{coord}")
                model.addConstr(
                    sep == positions[sat][t][coord] - initial_state["Deb1"]["position"][coord]
                )
                model.addConstr(sep >= d_safe/3)
    
    # Objective: Minimize total fuel
    model.setObjective(
        gp.quicksum(delta[sat, t] for sat in satellites for t in time_steps[:-1]),
        GRB.MINIMIZE
    )
    
    # Solve and print results
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("\nLP Optimal Solution Found")
        print(f"Total Fuel: {model.ObjVal:.2f} kg")
        # Add code to extract positions/thrust vectors
    else:
        print("No LP solution found")

# # ---------------------------
# # MILP Model (Pulsed Thrust)
# # ---------------------------
# def solve_milp_model():
#     model = gp.Model("SatelliteMILP")
    
#     # Decision Variables
#     u = {}  # Thrust vectors
#     z = {}  # Thruster activation binaries
#     delta = {}  # Fuel consumption
    
#     for sat in satellites:
#         for t in time_steps[:-1]:  # No thrust at final step
#             # Thruster activation binary
#             z[sat, t] = model.addVar(
#                 vtype=GRB.BINARY, 
#                 name=f"z_{sat}_{t}"
#             )
            
#             # Thrust vector components (-F_max*z to F_max*z)
#             u[sat, t] = model.addVars(
#                 3, 
#                 lb=-F_max*z[sat, t], 
#                 ub=F_max*z[sat, t], 
#                 name=f"u_{sat}_{t}"
#             )
            
#             # Fuel consumption (big-M constraints)
#             delta[sat, t] = model.addVar(name=f"delta_{sat}_{t}")
#             for coord in range(3):
#                 abs_u = model.addVar(name=f"abs_u_{sat}_{t}_{coord}")
#                 model.addGenConstrAbs(abs_u, u[sat, t][coord])
#                 model.addConstr(
#                     delta[sat, t] >= c_fuel * abs_u - M*(1 - z[sat, t])
#                 )
#                 model.addConstr(
#                     delta[sat, t] <= M * z[sat, t]
#                 )
    
#     # Add orbital dynamics and position variables
#     positions = {}
#     for sat in satellites:
#         p, v = add_orbital_dynamics(model, sat, time_steps, dt, m_i)
#         positions[sat] = p
    
#     # Collision Avoidance Constraints (same as LP)
#     # ... [Identical to LP version] ...
    
#     # Additional Constraints: Max 2 activations per satellite
#     for sat in satellites:
#         model.addConstr(
#             gp.quicksum(z[sat, t] for t in time_steps[:-1]) <= 2,
#             name=f"max_activations_{sat}"
#         )
    
#     # Objective: Minimize total fuel
#     model.setObjective(
#         gp.quicksum(delta[sat, t] for sat in satellites for t in time_steps[:-1]),
#         GRB.MINIMIZE
#     )
    
#     # Solve and print results
#     model.Params.TimeLimit = 300  # 5 minutes
#     model.optimize()
    
#     if model.status == GRB.OPTIMAL:
#         print("\nMILP Optimal Solution Found")
#         print(f"Total Fuel: {model.ObjVal:.2f} kg")
#         # Add code to extract activation patterns
#     else:
#         print("No MILP solution found")

# ---------------------------
# Run Both Models
# ---------------------------
if __name__ == "__main__":
    print("Solving LP Model...")
    solve_lp_model()
    
    # print("\nSolving MILP Model...")
    # solve_milp_model()