
import numpy as np
import matplotlib.pyplot as plt
import json

def histogram_from_trajectory_slice(
    trajectory_slice: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    bins: int = 100
) -> tuple[np.ndarray, list[float]]:
    import numpy as np

    positions = trajectory_slice.reshape(-1, 2)
    x, y = positions[:, 0], positions[:, 1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    hist = hist.T  # Transpose so that [y,x] matches image orientation
    hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return hist_norm, extent

def plot_density_map_from_histogram(
    hist: np.ndarray,
    extent: list[float],
    gamma: float,
    square: bool,
    title_suffix: str = ""
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(hist, origin='lower', cmap='Greys', extent=extent, interpolation='nearest')
    ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)
    ax.set_xlabel('x position (μm)')
    ax.set_ylabel('y position (μm)')
    ax.set_title(f'Ion Density Map: γ = {gamma:.6f} {title_suffix}')
    if square:
        ax.set_aspect('equal')
        
    else:
        ax.set_aspect('auto')
        
    fig.colorbar(im, ax=ax, label='Normalized density')
    return fig

def find_xy_bounds(traj_file, x_scaling_factor = 1., y_scaling_factor = 1.3):
    with open(traj_file, 'r') as f:
        traj = np.array(json.load(f)).reshape(-1, 2)
    x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
    y_min, y_max = traj[:, 1].min(), traj[:, 1].max()
    x_min *= x_scaling_factor
    x_max *= x_scaling_factor
    y_min *= y_scaling_factor
    y_max *= y_scaling_factor
    return x_min, x_max, y_min, y_max
