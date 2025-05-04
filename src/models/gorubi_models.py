import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from src.plotting.plot_trajectories import plot_satellite_trajectories

# ---------------------------
# Sample Data Initialization
# ---------------------------
satellites = ["Sat1", "Sat2"]
time_steps = list(range(20))
dt = 20  # Time step (seconds)
F_max = 500000  # Thrust (N)
c_fuel = 0.00001
m_i = 0.001  # Satellite mass (kg)
d_safe = 100000  # Safe distance (meters)
M = 1e100  # Big-M constant

initial_state = {
    "Sat1": {
        "position": [1000, 2000, 0],
        "velocity": [1000, 0 , 0]
    },
    "Sat2": {
        "position": [15000, 20000, 0],
        "velocity": [1000, 1000, 0]
    }
}

# ---------------------------
# Helper Function with Velocity Initialization
# ---------------------------
def add_orbital_dynamics(model, sat, time_steps, dt, mass, u, initial_pos, initial_vel):
    p = {t: model.addVars(3, name=f"p_{sat}_{t}") for t in time_steps}
    v = {t: model.addVars(3, name=f"v_{sat}_{t}") for t in time_steps}
    
    for coord in range(3):
        model.addConstr(p[0][coord] == initial_pos[sat][coord])
        model.addConstr(v[0][coord] == initial_vel[sat][coord])
    
    for t in time_steps[:-1]:
        for coord in range(3):
            model.addConstr(
                p[t+1][coord] == p[t][coord] + v[t][coord] * dt + 0.5 * (u[sat, t][coord] / mass) * dt**2
            )
            model.addConstr(
                v[t+1][coord] == v[t][coord] + (u[sat, t][coord] / mass) * dt
            )
    return p, v

# ---------------------------
# LP Model
# ---------------------------
def solve_lp_model():
    model = gp.Model("SatelliteLP")
    u = {}
    delta = {}

    for sat in satellites:
        for t in time_steps[:-1]:
            u[sat, t] = model.addVars(3, lb=-F_max, ub=F_max, name=f"u_{sat}_{t}")
            delta[sat, t] = model.addVar(name=f"delta_{sat}_{t}")
            for coord in range(3):
                abs_u = model.addVar(name=f"abs_u_{sat}_{t}_{coord}")
                model.addGenConstrAbs(abs_u, u[sat, t][coord])
                model.addConstr(delta[sat, t] >= c_fuel * abs_u)

    initial_pos = {sat: initial_state[sat]["position"] for sat in satellites}
    initial_vel = {sat: initial_state[sat]["velocity"] for sat in satellites}
    
    positions = {}
    velocities = {}
    for sat in satellites:
        p, v = add_orbital_dynamics(model, sat, time_steps, dt, m_i, u, initial_pos, initial_vel)
        positions[sat] = p
        velocities[sat] = v

    for t in time_steps:
        for (i, j) in combinations(satellites, 2):
            b_x = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{t}_x")
            b_y = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{t}_y")
            b_z = model.addVar(vtype=GRB.BINARY, name=f"b_{i}_{j}_{t}_z")
            
            model.addConstr(positions[i][t][0] - positions[j][t][0] >= d_safe - M * (1 - b_x))
            model.addConstr(positions[j][t][0] - positions[i][t][0] >= d_safe - M * (1 - b_x))
            model.addConstr(positions[i][t][1] - positions[j][t][1] >= d_safe - M * (1 - b_y))
            model.addConstr(positions[j][t][1] - positions[i][t][1] >= d_safe - M * (1 - b_y))
            model.addConstr(positions[i][t][2] - positions[j][t][2] >= d_safe - M * (1 - b_z))
            model.addConstr(positions[j][t][2] - positions[i][t][2] >= d_safe - M * (1 - b_z))
            model.addConstr(b_x + b_y + b_z >= 1)

    model.setObjective(gp.quicksum(delta[sat, t] for sat in satellites for t in time_steps[:-1]), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        plot_satellite_trajectories(model)
    else:
        print("Model infeasible. Check constraints.")

if __name__ == "__main__":
    solve_lp_model()
