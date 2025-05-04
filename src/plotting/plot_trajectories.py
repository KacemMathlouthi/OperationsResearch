import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import matplotlib.animation as animation

# Updated to accept time_steps explicitly
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
    # Extract data
    positions = {}
    thrusts = {}
    p_pattern = re.compile(r'p_(\w+)_(\d+)\[(\d+)\]')
    u_pattern = re.compile(r'u_(\w+)_(\d+)\[(\d+)\]')
    
    for var in model.getVars():
        pm = p_pattern.match(var.VarName)
        if pm:
            sat, t, coord = pm.groups()
            t, coord = int(t), int(coord)
            positions.setdefault(sat, {}).setdefault(t, [0.0,0.0,0.0])[coord] = var.X
            continue
        um = u_pattern.match(var.VarName)
        if um:
            sat, t, coord = um.groups()
            t, coord = int(t), int(coord)
            thrusts.setdefault(sat, {}).setdefault(t, [0.0,0.0,0.0])[coord] = var.X

    sats = sorted(positions.keys())

    # prepare figure
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Satellite Trajectories and Thrust Animation')

    # create plot objects
    traj_lines = {sat: ax.plot([], [], [], label=f'{sat} traj')[0] for sat in sats}
    end_markers = {sat: ax.plot([], [], [], marker='o', linestyle='')[0] for sat in sats}
    quiver_objs = {sat: None for sat in sats}

    # set axis limits based on all data
    all_x = [positions[s][t][0] for s in sats for t in positions[s]]
    all_y = [positions[s][t][1] for s in sats for t in positions[s]]
    all_z = [positions[s][t][2] for s in sats for t in positions[s]]
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))
    ax.legend()

    def update(frame):
        # remove old quivers
        for sat in sats:
            if quiver_objs[sat] is not None:
                quiver_objs[sat].remove()
        
        for sat in sats:
            # trajectory up to frame
            xs = [positions[sat][t][0] for t in time_steps if t <= frame]
            ys = [positions[sat][t][1] for t in time_steps if t <= frame]
            zs = [positions[sat][t][2] for t in time_steps if t <= frame]
            traj_lines[sat].set_data(xs, ys)
            traj_lines[sat].set_3d_properties(zs)
            # end marker
            if xs:
                end_markers[sat].set_data([xs[-1]], [ys[-1]])
                end_markers[sat].set_3d_properties([zs[-1]])
            # thrust arrow at this frame
            ux, uy, uz = thrusts.get(sat, {}).get(frame, [0.0,0.0,0.0])
            quiver_objs[sat] = ax.quiver(
                xs[-1], ys[-1], zs[-1], ux, uy, uz,
                length=1e3, normalize=True
            )
        return list(traj_lines.values()) + list(end_markers.values()) + [q for q in quiver_objs.values()]

    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=interval, blit=False)
    plt.show()
