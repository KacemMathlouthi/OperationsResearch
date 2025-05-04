import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import gurobipy as gp

def plot_satellite_trajectories(model):
    """
    Plots 3D trajectories of satellites using position variables from a solved Gurobi model.
    
    Args:
        model (gp.Model): Solved Gurobi model containing satellite position variables 
                          named as `p_{satellite}_{time}[coord]` (e.g., `p_Sat1_0[0]`).
    """
    positions = {}
    pattern = re.compile(r'p_(\w+)_(\d+)\[(\d+)\]')  # Matches p_Sat1_0[0]

    for var in model.getVars():
        match = pattern.match(var.VarName)
        if not match:
            continue

        sat, t, coord = match.groups()
        t = int(t)
        coord = int(coord)
        value = var.X

        if sat not in positions:
            positions[sat] = {}
        if t not in positions[sat]:
            positions[sat][t] = [0.0, 0.0, 0.0]
        
        positions[sat][t][coord] = value

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for sat in positions:
        times = sorted(positions[sat].keys())
        x = [positions[sat][t][0] for t in times]
        y = [positions[sat][t][1] for t in times]
        z = [positions[sat][t][2] for t in times]
        
        ax.plot(x, y, z, marker='o', linestyle='-', label=sat)
        ax.text(x[-1], y[-1], z[-1], f'{sat} (end)', color='black')
    # Print final positions and velocities
    for sat in positions:
        final_time = max(positions[sat].keys())
        final_position = positions[sat][final_time]
        print(f"Satellite: {sat}")
        print(f"  Final Position (X, Y, Z): {final_position}")
        if final_time > 0:
            prev_position = positions[sat][final_time - 1]
            velocity = [(final_position[i] - prev_position[i]) for i in range(3)]
            print(f"  Approx. Velocity (X, Y, Z): {velocity}")
        else:
            print("  Velocity: Not available (only one time step)")


    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Satellite Trajectories')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
