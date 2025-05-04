import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import combinations
from src.plotting.plot_trajectories import plot_satellite_trajectories
# ---------------------------
# Sample Data Initialization
# ---------------------------
# Adjusted parameters
satellites = ["Sat1", "Sat2"]
time_steps = [0, 1, 2]
dt = 10  # Time step (seconds)
F_max = 5000  # Increased thrust (N)
c_fuel = 0.001
m_i = 1000  # Satellite mass (kg)
d_safe = 5000  # Reduced safe distance (meters)
M = 1e3  # Large-M value (adjust based on position bounds)

# Initial positions with sufficient separation
initial_position = {
    "Sat1": [1000, 2000, 3000],
    "Sat2": [15000, 20000, 25000],
    "Deb1": [10000, 10000, 10000]  # Far from both satellites
}


# Initial positions and velocities (3D: x,y,z)
initial_state = {
    "Sat1": {
        "position": [1000, 2000, 3000],
        "velocity": [0, 60, 0]  # ~Low Earth Orbit velocity
    },
    "Sat2": {
        "position": [15000, 20000, 25000],
        "velocity": [0, 70, 0]
    },
    "Deb1": {
        "position": [10000, 10000, 10000],
        "velocity": [0, 76, 0]
    }
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
    
    for t in time_steps:
        # --------------------------------------------
        # Satellite-to-Satellite Collision Avoidance
        # --------------------------------------------
        for (i, j) in combinations(satellites, 2):
            # Binary variables for each coordinate
            b_x = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{t}_x")
            b_y = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{t}_y")
            b_z = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{t}_z")
            
            # X-coordinate separation (either direction >= d_safe)
            model.addConstr(
                positions[i][t][0] - positions[j][t][0] >= d_safe - M * (1 - b_x),
                name=f"sep_x_positive_{i}_{j}_{t}"
            )
            model.addConstr(
                positions[j][t][0] - positions[i][t][0] >= d_safe - M * (1 - b_x),
                name=f"sep_x_negative_{i}_{j}_{t}"
            )
            
            # Y-coordinate separation
            model.addConstr(
                positions[i][t][1] - positions[j][t][1] >= d_safe - M * (1 - b_y),
                name=f"sep_y_positive_{i}_{j}_{t}"
            )
            model.addConstr(
                positions[j][t][1] - positions[i][t][1] >= d_safe - M * (1 - b_y),
                name=f"sep_y_negative_{i}_{j}_{t}"
            )
            
            # Z-coordinate separation
            model.addConstr(
                positions[i][t][2] - positions[j][t][2] >= d_safe - M * (1 - b_z),
                name=f"sep_z_positive_{i}_{j}_{t}"
            )
            model.addConstr(
                positions[j][t][2] - positions[i][t][2] >= d_safe - M * (1 - b_z),
                name=f"sep_z_negative_{i}_{j}_{t}"
            )
            
            # At least one coordinate must satisfy separation
            model.addConstr(b_x + b_y + b_z >= 1, name=f"sep_logic_{i}_{j}_{t}")

        # --------------------------------------------
        # Satellite-to-Debris Collision Avoidance
        # --------------------------------------------
        deb1_pos = initial_position["Deb1"]  # Use consistent initial position
        for sat in satellites:
            # Binary variables for each coordinate
            b_x = model.addVar(vtype=GRB.BINARY, name=f"b_{sat}_Deb1_{t}_x")
            b_y = model.addVar(vtype=GRB.BINARY, name=f"b_{sat}_Deb1_{t}_y")
            b_z = model.addVar(vtype=GRB.BINARY, name=f"b_{sat}_Deb1_{t}_z")
            
            # X-coordinate separation
            model.addConstr(
                positions[sat][t][0] - deb1_pos[0] >= d_safe - M * (1 - b_x),
                name=f"sep_x_positive_{sat}_Deb1_{t}"
            )
            model.addConstr(
                deb1_pos[0] - positions[sat][t][0] >= d_safe - M * (1 - b_x),
                name=f"sep_x_negative_{sat}_Deb1_{t}"
            )
            
            # Y-coordinate separation
            model.addConstr(
                positions[sat][t][1] - deb1_pos[1] >= d_safe - M * (1 - b_y),
                name=f"sep_y_positive_{sat}_Deb1_{t}"
            )
            model.addConstr(
                deb1_pos[1] - positions[sat][t][1] >= d_safe - M * (1 - b_y),
                name=f"sep_y_negative_{sat}_Deb1_{t}"
            )
            
            # Z-coordinate separation
            model.addConstr(
                positions[sat][t][2] - deb1_pos[2] >= d_safe - M * (1 - b_z),
                name=f"sep_z_positive_{sat}_Deb1_{t}"
            )
            model.addConstr(
                deb1_pos[2] - positions[sat][t][2] >= d_safe - M * (1 - b_z),
                name=f"sep_z_negative_{sat}_Deb1_{t}"
            )
            
            model.addConstr(b_x + b_y + b_z >= 1, name=f"sep_logic_{sat}_Deb1_{t}")
        
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
        plot_satellite_trajectories(model)
    else:
        model.computeIIS()
        model.write("infeasible.ilp")
        print("Model is infeasible. Check 'infeasible.ilp'.")
        print("No LP solution found")



if __name__ == "__main__":
    print("Solving LP Model...")
    solve_lp_model()
    
