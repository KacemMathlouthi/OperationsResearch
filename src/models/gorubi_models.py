import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from src.plotting.plot_trajectories import animate_satellite_trajectories

# ---------------------------
# Sample Data Initialization
# ---------------------------

time_steps = list(range(30))
dt = 10.0            # Time step (seconds)
F_max = 1e6          # Thrust (N)
c_fuel = 1e-5        # Fuel cost factor
m_i = 0.01           # Satellite mass (kg)
d_safe = 1000.0         # Safe distance disabled

initial_state = {
    "Sat1": {"position": [20000.0, 0.0, 0.0],    "velocity": [-100.0, 0.0, 0.0]},
    "Sat2": {"position": [-10000.0, 17320.51, 0.0],"velocity": [50.0, -86.6025, 0.0]},
    "Sat3": {"position": [-10000.0, -17320.51, 0.0],"velocity": [50.0, 86.6025, 0.0]},
}
satellites = list(initial_state.keys())

# ---------------------------
# Add orbital dynamics
# ---------------------------
def add_orbital_dynamics(model, sat, time_steps, dt, mass, u, initial_pos, initial_vel):
    # position and velocity variables (allow negative values)
    p = {t: model.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_{sat}_{t}") for t in time_steps}
    v = {t: model.addVars(3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"v_{sat}_{t}") for t in time_steps}
    # initial conditions
    for coord in range(3):
        model.addConstr(p[0][coord] == initial_pos[sat][coord])
        model.addConstr(v[0][coord] == initial_vel[sat][coord])
    # dynamics
    for t in time_steps[:-1]:
        for coord in range(3):
            model.addConstr(
                p[t+1][coord] == p[t][coord] + v[t][coord]*dt + 0.5*(u[sat,t][coord]/mass)*dt*dt
            )
            model.addConstr(
                v[t+1][coord] == v[t][coord] + (u[sat,t][coord]/mass)*dt
            )
    return p, v

# ---------------------------
# Build and solve model
# ---------------------------
def solve_lp_model():
    model = gp.Model("SatelliteLP")
    # control and fuel variables
    u = {}
    delta = {}
    for sat in satellites:
        for t in time_steps[:-1]:
            u[sat,t] = model.addVars(3, lb=-F_max, ub=F_max, name=f"u_{sat}_{t}")
            delta[sat,t] = model.addVar(lb=0.0, name=f"delta_{sat}_{t}")
            # link fuel to thrust
            for k in range(3):
                abs_u = model.addVar(lb=0.0, name=f"abs_u_{sat}_{t}_{k}")
                model.addGenConstrAbs(abs_u, u[sat,t][k])
                model.addConstr(delta[sat,t] >= c_fuel*abs_u)
    # dynamics
    initial_pos = {s: initial_state[s]["position"] for s in satellites}
    initial_vel = {s: initial_state[s]["velocity"] for s in satellites}
    positions, velocities = {}, {}
    for sat in satellites:
        p,v = add_orbital_dynamics(model, sat, time_steps, dt, m_i, u, initial_pos, initial_vel)
        positions[sat], velocities[sat] = p, v
    # no collision constraints when d_safe=0

    # objective: minimize total fuel
    model.setObjective(
        gp.quicksum(delta[s,t] for s in satellites for t in time_steps[:-1]), GRB.MINIMIZE
    )
    model.optimize()
    # debug infeasibility
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write('infeasible.ilp')
        raise Exception('Model infeasible; IIS written')
    # animate
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        animate_satellite_trajectories(model, time_steps)

if __name__ == '__main__':
    solve_lp_model()
