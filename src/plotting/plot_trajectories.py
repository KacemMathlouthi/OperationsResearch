import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gurobipy as gp

def plot_satellite_trajectories(model):
    """
    Plots 3D trajectories of satellites using position variables from a solved Gurobi model.
    
    Args:
        model (gp.Model): Solved Gurobi model containing satellite position variables 
                          named as `p_{satellite}_{time}_{coordinate}` (e.g., `p_Sat1_0_0`).
    """
    # Extract position data from model variables
    positions = {}
    for var in model.getVars():
        if var.VarName.startswith('p_'):
            parts = var.VarName.split('_')
            if len(parts) != 4:
                continue  # Skip incorrectly named variables
            
            sat = parts[1]
            t = int(parts[2])
            coord = int(parts[3])
            value = var.X  # Optimized value
            
            # Initialize satellite/time if not exists
            if sat not in positions:
                positions[sat] = {}
            if t not in positions[sat]:
                positions[sat][t] = [0.0, 0.0, 0.0]  # x, y, z
            
            # Update coordinate value
            positions[sat][t][coord] = value

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each satellite's trajectory
    for sat in positions:
        # Sort time steps and extract coordinates
        times = sorted(positions[sat].keys())
        x = [positions[sat][t][0] for t in times]
        y = [positions[sat][t][1] for t in times]
        z = [positions[sat][t][2] for t in times]
        
        ax.plot(x, y, z, marker='o', linestyle='-', label=sat)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Satellite Trajectories')
    plt.legend()
    plt.show()