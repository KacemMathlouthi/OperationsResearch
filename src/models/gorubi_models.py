import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import matplotlib.animation as animation

def animate_satellite_trajectories(model, time_steps, interval=200):
    """
    Creates a 3D animated plot of satellite trajectories and thrust vectors.

    Args:
        model (gp.Model): Solved Gurobi model containing:
            - position variables named `p_{sat}_{t}[coord]`
            - thrust variables named `u_{sat}_{t}[coord]`
        time_steps (list of int): List of time step indices used in the optimization.
        interval (int): Delay between frames in milliseconds.
    """
    # Extract data containers
    positions = {}
    thrusts = {}
    # Patterns to match variable names
    p_pattern = re.compile(r'p_(\w+)_(\d+)\[(\d+)\]')
    u_pattern = re.compile(r'u_(\w+)_(\d+)\[(\d+)\]')

    # Populate positions and thrusts dictionaries
    for var in model.getVars():
        # Position variables
        pm = p_pattern.match(var.VarName)
        if pm:
            sat, t_str, coord_str = pm.groups()
            t = int(t_str)
            coord = int(coord_str)
            positions.setdefault(sat, {}).setdefault(t, [0.0, 0.0, 0.0])[coord] = var.Xn
            continue
        # Thrust variables
        um = u_pattern.match(var.VarName)
        if um:
            sat, t_str, coord_str = um.groups()
            t = int(t_str)
            coord = int(coord_str)
            thrusts.setdefault(sat, {}).setdefault(t, [0.0, 0.0, 0.0])[coord] = var.Xn

    sats = sorted(positions.keys())

    # Prepare 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Satellite Trajectories and Thrust Animation')

    # Plot objects for trajectories, end markers, and quivers
    traj_lines = {sat: ax.plot([], [], [], label=f'{sat} traj')[0] for sat in sats}
    end_markers = {sat: ax.plot([], [], [], marker='o', linestyle='')[0] for sat in sats}
    quiver_objs = {sat: None for sat in sats}

    # Set axis limits based on all positions
    all_x = [positions[s][t][0] for s in sats for t in positions[s]]
    all_y = [positions[s][t][1] for s in sats for t in positions[s]]
    all_z = [positions[s][t][2] for s in sats for t in positions[s]]
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))
    ax.legend()

    def update(frame):
        # Remove previous quivers
        for sat in sats:
            if quiver_objs[sat] is not None:
                quiver_objs[sat].remove()

        # Update each satellite's trajectory, marker, and thrust arrow
        for sat in sats:
            # Trajectory up to current frame
            xs = [positions[sat][t][0] for t in time_steps if t <= frame]
            ys = [positions[sat][t][1] for t in time_steps if t <= frame]
            zs = [positions[sat][t][2] for t in time_steps if t <= frame]
            traj_lines[sat].set_data(xs, ys)
            traj_lines[sat].set_3d_properties(zs)

            # End marker at last position
            if xs:
                end_markers[sat].set_data([xs[-1]], [ys[-1]])
                end_markers[sat].set_3d_properties([zs[-1]])

            # Thrust arrow at this frame
            ux, uy, uz = thrusts.get(sat, {}).get(frame, [0.0, 0.0, 0.0])
            # Draw new quiver
            quiver_objs[sat] = ax.quiver(
                xs[-1], ys[-1], zs[-1], ux, uy, uz,
                length=1e3, normalize=True
            )

        # Return artists to update
        artists = list(traj_lines.values()) + list(end_markers.values())
        artists += [q for q in quiver_objs.values() if q is not None]
        return artists

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=time_steps, interval=interval, blit=False
    )
    ani.save("satellite_trajectories.gif", writer='imagemagick', fps=5)
    plt.show()
# ---------------------------
# SCALED DATA INITIALIZATION
# ---------------------------
# Units:
#  • distance unit = 10 km
#  • time unit = 10 s
#  • velocity unit = 1 km/s
#  • thrust unit = 1 kN
#  • mass unit = 0.01 kg

time_steps = list(range(20))
dt = 10.0             # 10 s
F_max = 10.0        # 100 kN
c_fuel = 100     # same relative cost
m_i = 1000.0           # 0.01 kg
# safety distance scaled: 0.00001 units = 0.1 m
# we apply collision avoidance only from t=1 onward to respect initial positions
d_safe = 1

initial_state = {
    "Sat1": { "position": [20000/10000,     0.0,   0.0],   "velocity": [-100/1000,   0.0,    0.0] },
    "Sat2": { "position": [-10000/10000, 17320.51/10000, 0.0], "velocity": [ 50/1000, -86.6025/1000, 0.0] },
    "Sat3": { "position": [-10000/10000,-17320.51/10000, 0.0], "velocity": [ 50/1000,  86.6025/1000, 0.0] },
}
satellites = list(initial_state.keys())

# ------------------------------------
# PRECOMPUTE NOMINAL (UNCORRECTED) TRAJECTORIES
# ------------------------------------
# Nominal case: no thrust (u=0)
nominal_positions = {sat: {0: initial_state[sat]['position'][:] } for sat in satellites}
for sat in satellites:
    pos = nominal_positions[sat]
    vel = initial_state[sat]['velocity']
    for t in time_steps[:-1]:
        # constant velocity motion
        next_pos = [pos[t][i] + vel[i] * dt for i in range(3)]
        pos[t+1] = next_pos

# ------------------------------------
# ORBITAL DYNAMICS FUNCTION
# ------------------------------------
def add_orbital_dynamics(model, sat, time_steps, dt, mass, u, initial_pos, initial_vel):
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
# BUILD & SOLVE WITH TRACKING PENALTY
# ---------------------------

def solve_lp_model():
    model = gp.Model("SatelliteLP_WithTracking")

    # decision vars: thrust u, fuel cost delta
    u = {}
    delta = {}
    for sat in satellites:
        for t in time_steps[:-1]:
            u[sat, t] = model.addVars(3, lb=-F_max, ub=F_max, name=f"u_{sat}_{t}")
            delta[sat, t] = model.addVar(lb=0.0, name=f"delta_{sat}_{t}")
            for k in range(3):
                abs_u = model.addVar(lb=0.0, name=f"abs_u_{sat}_{t}_{k}")
                model.addGenConstrAbs(abs_u, u[sat, t][k], name=f"absConstr_{sat}_{t}_{k}")
                model.addConstr(delta[sat, t] >= c_fuel * abs_u, name=f"fuelLink_{sat}_{t}_{k}")

    # dynamics and collect position vars
    initial_pos = {s: initial_state[s]["position"] for s in satellites}
    initial_vel = {s: initial_state[s]["velocity"] for s in satellites}
    positions = {}
    for sat in satellites:
        p, v = add_orbital_dynamics(model, sat, time_steps, dt, m_i, u, initial_pos, initial_vel)
        positions[sat] = p

    # collision avoidance as before
    for t in time_steps[1:]:
        for i, j in combinations(satellites, 2):
            dx = model.addVar(name=f"dx_{i}_{j}_{t}")
            dy = model.addVar(name=f"dy_{i}_{j}_{t}")
            dz = model.addVar(name=f"dz_{i}_{j}_{t}")
            model.addConstr(dx == positions[i][t][0] - positions[j][t][0], name=f"con_dx_{i}_{j}_{t}")
            model.addConstr(dy == positions[i][t][1] - positions[j][t][1], name=f"con_dy_{i}_{j}_{t}")
            model.addConstr(dz == positions[i][t][2] - positions[j][t][2], name=f"con_dz_{i}_{j}_{t}")
            absx = model.addVar(lb=0.0, name=f"absx_{i}_{j}_{t}")
            absy = model.addVar(lb=0.0, name=f"absy_{i}_{j}_{t}")
            absz = model.addVar(lb=0.0, name=f"absz_{i}_{j}_{t}")
            model.addGenConstrAbs(absx, dx, name=f"absxConstr_{i}_{j}_{t}")
            model.addGenConstrAbs(absy, dy, name=f"absyConstr_{i}_{j}_{t}")
            model.addGenConstrAbs(absz, dz, name=f"abszConstr_{i}_{j}_{t}")
            bx = model.addVar(vtype=GRB.BINARY, name=f"bx_{i}_{j}_{t}")
            by = model.addVar(vtype=GRB.BINARY, name=f"by_{i}_{j}_{t}")
            bz = model.addVar(vtype=GRB.BINARY, name=f"bz_{i}_{j}_{t}")
            model.addConstr(absx >= d_safe * bx, name=f"safe_x_{i}_{j}_{t}")
            model.addConstr(absy >= d_safe * by, name=f"safe_y_{i}_{j}_{t}")
            model.addConstr(absz >= d_safe * bz, name=f"safe_z_{i}_{j}_{t}")
            model.addConstr(bx + by + bz >= 1, name=f"sep_sum_{i}_{j}_{t}")

    # tracking penalty: absolute deviation from nominal
    tracking_dev = {}
    for sat in satellites:
        for t in time_steps:
            for coord in range(3):
                dev = model.addVar(lb=0.0, name=f"dev_{sat}_{t}_{coord}")
                tracking_dev[sat, t, coord] = dev
                # deviation = p - p_nom
                p_var = positions[sat][t][coord]
                p_nom = nominal_positions[sat][t][coord]
                # p_var - p_nom <= dev and -(p_var - p_nom) <= dev
                model.addConstr(p_var - p_nom <= dev, name=f"dev_pos_{sat}_{t}_{coord}")
                model.addConstr(p_nom - p_var <= dev, name=f"dev_neg_{sat}_{t}_{coord}")

    # objective: fuel + tracking deviation
    obj = gp.quicksum(delta[s, t] for s in satellites for t in time_steps[:-1]) \
        + gp.quicksum(tracking_dev[s, t, c] for s in satellites for t in time_steps for c in range(3))
    model.setObjective(obj, GRB.MINIMIZE)

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
