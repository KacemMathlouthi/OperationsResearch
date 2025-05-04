import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import gurobipy as gp

def plot_satellite_trajectories(model):
    """
    Plots 3D trajectories of satellites and thrust vectors using position and control variables from a solved Gurobi model.
    Args:
        model (gp.Model): Solved Gurobi model containing
            - position variables named `p_{sat}_{t}[coord]`
            - thrust variables named `u_{sat}_{t}[coord]`
    """
    # Extract positions
    positions = {}
    p_pattern = re.compile(r'p_(\w+)_(\d+)\[(\d+)\]')
    # Extract thrusts
    thrusts = {}
    u_pattern = re.compile(r'u_(\w+)_(\d+)\[(\d+)\]')

    for var in model.getVars():
        # positions
        pm = p_pattern.match(var.VarName)
        if pm:
            sat, t, coord = pm.groups()
            t, coord = int(t), int(coord)
            positions.setdefault(sat, {}).setdefault(t, [0.0,0.0,0.0])[coord] = var.X
            continue
        # thrusts
        um = u_pattern.match(var.VarName)
        if um:
            sat, t, coord = um.groups()
            t, coord = int(t), int(coord)
            thrusts.setdefault(sat, {}).setdefault(t, [0.0,0.0,0.0])[coord] = var.X

    # Plot
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')

    for sat in sorted(positions.keys()):
        times = sorted(positions[sat].keys())
        x = [positions[sat][t][0] for t in times]
        y = [positions[sat][t][1] for t in times]
        z = [positions[sat][t][2] for t in times]
        # trajectory line
        ax.plot(x, y, z, marker='o', linestyle='-', label=f'{sat} traj')
        # thrust arrows
        for t in times[:-1]:
            # arrow at position at time t
            px, py, pz = positions[sat][t]
            ux, uy, uz = thrusts.get(sat, {}).get(t, [0,0,0])
            # scale factor for visibility
            scale = 10
            ax.quiver(px, py, pz, ux*scale, uy*scale, uz*scale, length=1.0, normalize=False)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Satellite Trajectories and Thrust Vectors')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
