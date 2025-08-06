# Written 2025-07-16
import numpy as np
import os
import json
from dataclasses import asdict
from simulation_module import SimulationConfig, SimulationState, SimulationRunner, SimulationIO, AnimationMaker  # Adjust import if needed
from histogram_workers import histogram_from_trajectory_slice, plot_density_map_from_histogram
import matplotlib.pyplot as plt
import re
import natural_units as units
from time import time
# import ijson

def run_single_for_dt(dt, config_dict, positions, run_time_us):
    start = time()
    config_dict = config_dict.copy()
    config_dict['dt'] = dt
    config_dict['num_steps'] = int(run_time_us / dt)
    config = SimulationConfig(**config_dict)

    state = SimulationState(config)
    state.positions = np.array(positions)
    state.initial_positions = np.array(positions)
    state.initialized = True

    runner = SimulationRunner(config, state)
    runner.run(verbose = True)

    temps = runner.state.temperatures
    if isinstance(temps, (list, np.ndarray)) and len(temps) >= 10:
        n = len(temps)
        start_idx = int(0.1 * n)
        early_temp_mK = units.theta_to_mK(np.mean(temps[:start_idx]))
        late_temp_mK = units.theta_to_mK(np.mean(temps[-start_idx:]))
    else:
        raise ValueError(f"Temperature trajectory is empty or too short. Length = {len(temps)}")

    
    return dt, early_temp_mK, late_temp_mK, time()-start


def process_single_density_map(
    traj_file: str,
    output_dir: str,
    base_config_dict: dict,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    index: int,
    square: bool,
    extension: str
):


    match = re.search(r"_(\d+\.\d+)_traj", os.path.basename(traj_file))
    gamma = float(match.group(1)) if match else 0.0
    filename = os.path.join(output_dir, f"frame_{index:03d}_gamma_{gamma:.6f}.{extension}")

    if os.path.exists(filename):
        print(f"Skipping frame {index:03d} (γ = {gamma:.6f}), image already exists.")
        return

    print(f"Processing {traj_file}")
    try:
        with open(traj_file, 'r') as f:
            traj = np.array(json.load(f))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading file {traj_file}: {e}")
        return

    config = SimulationConfig(**{**base_config_dict, 'g': gamma, 'num_steps': traj.shape[0]})
    state = SimulationState(config)
    state.trajectory = traj
    state.positions = traj[-1]
    state.initialized = True

    hist, extent = histogram_from_trajectory_slice(
        trajectory_slice=state.trajectory,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max
    )

    fig = plot_density_map_from_histogram(
        hist=hist,
        extent=extent,
        gamma=gamma,
        square = square,
        title_suffix=""
    )

    fig.savefig(filename)
    plt.close(fig)

    
def process_single_rolling_average_density_map(
    traj_file: str,
    output_dir: str,
    base_config_dict: dict,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    index: int,
    square: bool,
    extension: str,
    time_per_image: float = 100,
    overlapping_time: float = 0,
    full_histogram: bool = True,
    animate: bool = False,
):

    match = re.search(r"_(\d+\.\d+)_traj", os.path.basename(traj_file))
    gamma = float(match.group(1)) if match else 0.0
    
    try:
        with open(traj_file, 'r') as f:
            traj = np.array(json.load(f))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading file {traj_file}: {e}")
        return

    num_steps = traj.shape[0]
    dt = base_config_dict.get('dt', 0.001)  # μs
    
    assert num_steps * dt > time_per_image, "Time per image exceeds the length of full trajectory"
    
    window_size = int(time_per_image / dt)
    overlap = int(overlapping_time / dt)
    stride = window_size - overlap

    if stride <= 0:
        print("Invalid parameters: overlapping_time must be less than time_per_image.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, start in enumerate(range(0, num_steps - window_size + 1, stride)):
        filename = os.path.join(output_dir, f"rolling_frame_{index:03d}_window_{i:03d}_gamma_{gamma:.6f}.{extension}")
        if os.path.exists(filename):
            print(f"Skipping {filename}, already exists")
            continue
        end = start + window_size
        slice_traj = traj[start:end]
        hist, extent = histogram_from_trajectory_slice(slice_traj, x_min, x_max, y_min, y_max)
        fig = plot_density_map_from_histogram(hist, extent, gamma, square, title_suffix=f"(t = {start*dt:.1f}–{end*dt:.1f} μs)")
        fig.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")

    if full_histogram:
        full_filename = os.path.join(output_dir, f"rolling_frame_{index:03d}_FULL_gamma_{gamma:.6f}.{extension}")
        if os.path.exists(full_filename):
            print(f"Skipping {full_filename}, already exists")
        
        else:
            hist, extent = histogram_from_trajectory_slice(traj, x_min, x_max, y_min, y_max)
            fig = plot_density_map_from_histogram(hist, extent, gamma, square, title_suffix="(Full)")
            fig.savefig(full_filename)
            plt.close(fig)
            print(f"Saved full density map: {full_filename}")

    if animate:
        assert extension != "pdf", "You can't animate with extension pdf!! It must be png"
        try:
            anim_output_path = os.path.join(output_dir, f"rolling_animation_{index:03d}_gamma_{gamma:.6f}.mp4")
            if os.path.exists(anim_output_path):
                print(f"Skipping {anim_output_path}, already exists")
            else:
                AnimationMaker.make_gif_or_mp4_from_images(
                    image_folder = output_dir,
                    output_file = anim_output_path,
                    fps = 3,  # adjust as needed
                    reverse = False  # set to True if you want reverse playback
                )
                print(f"Animation saved to {anim_output_path}")
        except Exception as e:
            print(f"Failed to create animation: {e}")



def run_single_quench(gamma, base_config_dict, loadfile, output_dir):
    out_file = f"{output_dir}/{base_config_dict['N']}_{gamma:.14f}_traj_{base_config_dict['num_steps']}_steps.json"
    if os.path.exists(out_file):
        print(f"Skipping gamma = {gamma:.8f}, file exists.")
        return 

    print(f"Running gamma = {gamma:.8f}")

    config = SimulationConfig(**{**base_config_dict, 'g': gamma})
    state = SimulationState(config)

    state.positions = SimulationIO.load_positions(loadfile)
    state.initial_positions = state.positions.copy()
    state.initialized = True
    state.reset()

    x = state.positions[:, 0]
    y = state.positions[:, 1]
    g0 = base_config_dict['g']
    w0 = base_config_dict['w']
    m = base_config_dict['m']

    initial_PE = 0.5 * m * w0**2 * np.sum(x**2 + (g0 * y)**2)
    new_PE = 0.5 * m * w0**2 * np.sum(x**2 + (gamma * y)**2)
    new_w = w0 * np.sqrt(initial_PE / new_PE)

    state.config.w = new_w
    state.config.g = gamma

    runner = SimulationRunner(state.config, state)
    runner.run()
    SimulationIO.save_trajectory(state.trajectory, out_file)
#     return f"Written {out_file}"


def estimate_temperature(trajectory, dt, m, grainyness):
    n_steps, N, _ = trajectory.shape
    n_windows = n_steps // grainyness - 1
    temperatures = []
    times = []
    for i in range(n_windows):
        idx1 = i * grainyness
        idx2 = i * grainyness + 1
        delta_r = trajectory[idx2] - trajectory[idx1]
        v_avg = delta_r / dt
        kinetic_energy = 0.5 * m * np.sum(v_avg**2)
        T = kinetic_energy / (N * units.kB)
        temperatures.append(T)
        times.append(idx1 * dt)
    return np.array(times), np.array(temperatures)


def process_single_temperature_plot(
    traj_file: str,
    output_dir: str,
    base_config_dict: dict,
    grainyness: int,
    extension: str
):
    match = re.search(r"_(\d+\.\d+)_traj", os.path.basename(traj_file))
    gamma = float(match.group(1)) if match else 0.0
    outfile = os.path.join(output_dir, f"temperature_plot_{gamma:.6f}.{extension}")

    if os.path.exists(outfile):
        print(f"Skipping temperature plot for γ = {gamma:.6f}, image already exists.")
        return

    print(f"Generating temperature plot for {traj_file}")

    try:
        with open(traj_file, 'r') as f:
            traj = np.array(json.load(f))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading file {traj_file}: {e}", flush=True)
        return

    if traj.ndim != 3 or traj.shape[1:] != (base_config_dict['N'], 2):
        print(f"Skipping {traj_file}, unexpected shape {traj.shape}", flush=True)
        return

    times, temps = estimate_temperature(
        traj,
        base_config_dict['dt'],
        base_config_dict['m'],
        grainyness
    )

    temps_mK = units.theta_to_mK(temps)
    plt.figure(figsize=(8, 4))
    plt.plot(times, temps_mK, label="Temperature (mK)")
    plt.scatter(times[0], temps_mK[0], label=f"T₀ = {temps_mK[0]:.3f} mK")
    plt.legend()
    plt.xlabel("Time (μs)")
    plt.ylabel("Temperature (mK)")
    plt.title(f"Estimated Temperature vs Time (γ = {gamma:.4f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
