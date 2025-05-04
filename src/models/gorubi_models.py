import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from src.plotting.plot_trajectories import animate_satellite_trajectories

# ---------------------------
# SCALED DATA INITIALIZATION
# ---------------------------
# Units:
#  • distance unit = 10 km
#  • time unit = 10 s
#  • velocity unit = 1 km/s
#  • thrust unit = 1 kN
#  • mass unit = 0.01 kg

time_steps = list(range(30))
dt = 1.0             # 10 s
F_max = 100.0        # 100 kN
c_fuel = 1e-5        # same relative cost
m_i = 10.0           # 0.01 kg
# safety distance scaled: 0.00001 units = 0.1 m
# we apply collision avoidance only from t=1 onward to respect initial positions

d_safe = 0.00001

initial_state = {
    "Sat1": { "position": [20000/10000,     0.0,   0.0],   "velocity": [-100/1000,   0.0,    0.0] },
    "Sat2": { "position": [-10000/10000, 17320.51/10000, 0.0], "velocity": [ 50/1000, -86.6025/1000, 0.0] },
    "Sat3": { "position": [-10000/10000,-17320.51/10000, 0.0], "velocity": [ 50/1000,  86.6025/1000, 0.0] },
}
satellites = list(initial_state.keys())

# ---------------------------
# ORBITAL DYNAMICS
# ---------------------------
def add_orbital_dynamics(model, sat, time_steps, dt, mass, u, initial_pos, initial_vel):
    # create position and velocity vars: p[sat][t][coord], v likewise
    p = {t: model.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_{sat}_{t}") for t in time_steps}
    v = {t: model.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"v_{sat}_{t}") for t in time_steps}

    # initial conditions
    for coord in range(3):
        model.addConstr(p[0][coord] == initial_pos[sat][coord], name=f"init_p_{sat}_{coord}")
        model.addConstr(v[0][coord] == initial_vel[sat][coord], name=f"init_v_{sat}_{coord}")

    # dynamics for t=0..T-1
    for t in time_steps[:-1]:
        for coord in range(3):
            model.addConstr(
                p[t+1][coord] == p[t][coord]
                                 + v[t][coord] * dt
                                 + 0.5 * (u[sat, t][coord] / mass) * dt * dt,
                name=f"dyn_p_{sat}_{t}_{coord}"
            )
            model.addConstr(
                v[t+1][coord] == v[t][coord]
                                 + (u[sat, t][coord] / mass) * dt,
                name=f"dyn_v_{sat}_{t}_{coord}"
            )
    return p, v

# ---------------------------
# BUILD & SOLVE
# ---------------------------
def solve_lp_model():
    model = gp.Model("SatelliteLP")

    # decision vars: thrust u, fuel cost delta
    u = {}
    delta = {}
    for sat in satellites:
        for t in time_steps[:-1]:
            # 3D thrust vector
            u[sat, t] = model.addVars(3, lb=-F_max, ub=F_max, name=f"u_{sat}_{t}")
            # fuel consumption proxy
            delta[sat, t] = model.addVar(lb=0.0, name=f"delta_{sat}_{t}")
            # link delta to abs thrust via abs constraints
            for k in range(3):
                abs_u = model.addVar(lb=0.0, name=f"abs_u_{sat}_{t}_{k}")
                model.addGenConstrAbs(abs_u, u[sat, t][k], name=f"absConstr_{sat}_{t}_{k}")
                model.addConstr(delta[sat, t] >= c_fuel * abs_u, name=f"fuelLink_{sat}_{t}_{k}")

    # orbital dynamics
    initial_pos = {s: initial_state[s]["position"] for s in satellites}
    initial_vel = {s: initial_state[s]["velocity"] for s in satellites}
    positions = {}
    for sat in satellites:
        p, v = add_orbital_dynamics(model, sat, time_steps, dt, m_i, u, initial_pos, initial_vel)
        positions[sat] = p

    # collision avoidance: enforce only from t=1 to avoid initial overlap issues
    for t in time_steps[1:]:
        for i, j in combinations(satellites, 2):
            # difference vars
            dx = model.addVar(name=f"dx_{i}_{j}_{t}")
            dy = model.addVar(name=f"dy_{i}_{j}_{t}")
            dz = model.addVar(name=f"dz_{i}_{j}_{t}")
            model.addConstr(dx == positions[i][t][0] - positions[j][t][0], name=f"con_dx_{i}_{j}_{t}")
            model.addConstr(dy == positions[i][t][1] - positions[j][t][1], name=f"con_dy_{i}_{j}_{t}")
            model.addConstr(dz == positions[i][t][2] - positions[j][t][2], name=f"con_dz_{i}_{j}_{t}")
            # absolute differences
            absx = model.addVar(lb=0.0, name=f"absx_{i}_{j}_{t}")
            absy = model.addVar(lb=0.0, name=f"absy_{i}_{j}_{t}")
            absz = model.addVar(lb=0.0, name=f"absz_{i}_{j}_{t}")
            model.addGenConstrAbs(absx, dx, name=f"absxConstr_{i}_{j}_{t}")
            model.addGenConstrAbs(absy, dy, name=f"absyConstr_{i}_{j}_{t}")
            model.addGenConstrAbs(absz, dz, name=f"abszConstr_{i}_{j}_{t}")
            # binary separation
            bx = model.addVar(vtype=GRB.BINARY, name=f"bx_{i}_{j}_{t}")
            by = model.addVar(vtype=GRB.BINARY, name=f"by_{i}_{j}_{t}")
            bz = model.addVar(vtype=GRB.BINARY, name=f"bz_{i}_{j}_{t}")
            # enforce safe distance on at least one axis
            model.addConstr(absx >= d_safe * bx, name=f"safe_x_{i}_{j}_{t}")
            model.addConstr(absy >= d_safe * by, name=f"safe_y_{i}_{j}_{t}")
            model.addConstr(absz >= d_safe * bz, name=f"safe_z_{i}_{j}_{t}")
            model.addConstr(bx + by + bz >= 1, name=f"sep_sum_{i}_{j}_{t}")

    # objective: minimize total fuel proxy
    model.setObjective(gp.quicksum(delta[s, t] for s in satellites for t in time_steps[:-1]), GRB.MINIMIZE)

    # solve
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible.ilp")
        print("Model infeasible; IIS written to 'infeasible.ilp'.")
    else:
        print("Optimal solution found.")
        animate_satellite_trajectories(model, time_steps)


if __name__ == "__main__":
    solve_lp_model()
