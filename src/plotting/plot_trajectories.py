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