#Written 2025-07-25 by Jack Beda (jack.beda.ca).
###############################################################

'''
Most functions in parralel.py perform some kind of operation that is parralelized (i.e. it can run on multiple cores). For example, run_quench_series will run numerous ion Simulations in parralel to generate simulation trajectories. This file contains:

    run_convergence_test
        # I don't use this one very much, it runs many simulations using different timesteps for each one
        # so that we can compare how effective the simulation is on different timesteps
    
    plot_convergence_results
        # This is used in conjunction with run_convergence_test
    
    run_quench_series
        # This is the meat and potatos function. It chooses different values of gamma, runs the simulation, and saves
        # the trajectory
    
    generate_density_map_images_from_quench_folder
        # If you already have a bunch of trajectories in the quench_folder, this will generate generate density maps for you
    
    generate_rolling_average_density_map_images_from_quench_folder
        # Instead of simply getting density maps for every trajectory inside of quench_folder, it is often useful
        # to see density maps for slices of time (i.e. instead of the full t = 0 us to t = 1000 us, we might want
        # to see t = 0 us to t = 100 us and then t = 100 us to t = 200 us as density plots. This will let you do that
    
    generate_temperature_plots_from_quench_folder
        # Taking trajectories from quench_folder, we can compute kinetic energy over time as a loose proxy for temperature
        # this is a good one to call just to make sure things are going well. Even with laser cooling on, the temperature of 
        # the system has a propensity to increase over time. With laser cooling off it heats up very fast.
'''


import numpy as np
import os

from dataclasses import asdict
# ^Used to convert SimulationConfig (an object) into a dictionary for some purposes

from concurrent.futures import ProcessPoolExecutor, as_completed
# ^ProcessPoolExecutor does the parralel processing

from parralel_workers import run_single_quench, process_single_density_map, process_single_temperature_plot, run_single_for_dt, process_single_rolling_average_density_map, process_single_defect_plot
# Each of the 

import glob
import json
import re
from functools import partial
from simulation_module import SimulationConfig, SimulationState, SimulationIO
import matplotlib.pyplot as plt
from histogram_workers import find_xy_bounds

def run_convergence_test(base_config: SimulationConfig, loadfile: str, output_dir: str,
                         time_steps: list[float], run_time_us: float, num_workers: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    base_config_dict = asdict(base_config)
    IO = SimulationIO()
    positions = IO.load_positions(loadfile).tolist()  # Make JSON-safe for subprocesses

    print(f"🔁 Running {run_time_us} μs convergence test for various time steps...")
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_single_for_dt, dt, base_config_dict, positions, run_time_us): dt
            for dt in time_steps
        }

        for future in as_completed(futures):
            dt = futures[future]
            try:
                dt_val, early_temp, late_temp, real_runtime_s = future.result()
                print(f"✅ dt={dt_val:.5f} µs: Early Temp = {early_temp:.4f} mK, Late Temp = {late_temp:.4f} mK. Time taken = {real_runtime_s:.4f} s")
                results.append((dt_val, early_temp, late_temp))
            except Exception as e:
                print(f"❌ dt={dt:.5f} µs failed: {e}")

    results.sort(key=lambda x: x[0])
    results_path = os.path.join(output_dir, "convergence_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📊 Results saved to {results_path}")
    return results

def plot_defect_plots(final_positions_folder: str,
                      output_dir: str,
                      base_config: SimulationConfig,
                      num_workers: int = None,
                      tol = None,
                      square: bool = False,
                      extension: str = "png"):
    """
    Generate defect plots in parallel from <final_positions_folder> (e.g. 'my_quench_final_positions'),
    saving images to <output_dir>.
    """

    root = base_config.output_path
    src_dir = os.path.join(root, final_positions_folder)
    dst_dir = os.path.join(root, output_dir)
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(src_dir, "*.json")),
        key=lambda f: float(re.search(r"_(\d+\.\d+)_final_positions", os.path.basename(f)).group(1)) if re.search(r"_(\d+\.\d+)_final_positions", os.path.basename(f)) else -1.0,
        reverse=True
    )
    if not files:
        print(f"⚠️ No final-positions JSON files found in: {src_dir}")
        return

    base_config_dict = asdict(base_config)

    print(f"🧵 Generating defect plots from {len(files)} files in parallel...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_single_defect_plot,
                fp, dst_dir, base_config_dict, tol, square, extension
            )
            for fp in files
        ]
        for f in futures:
            f.result()  # surface exceptions

    print("✅ All defect plots generated.")
    
    
    
# After `results` is obtained from run_convergence_test
def plot_convergence_results(results, output_dir):
    dts = [r[0] for r in results]
    temp_diffs = [abs(r[2] - r[1]) for r in results]  # late_temp - early_temp

    plt.figure(figsize=(8, 6))
    plt.plot(dts, temp_diffs, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Timestep (µs)')
    plt.ylabel('ΔTemperature (mK) [Late - Early]')
    plt.title('Convergence of Temperature with Timestep')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plot_path = os.path.join(output_dir, "convergence_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Convergence plot saved to {plot_path}")

def run_quench_series(config, loadfile, output_dir,
                               g_start, g_end, g_step, num_workers = 4):
    full_output_dir = os.path.join(config.output_path,output_dir)
    os.makedirs(full_output_dir, exist_ok = True)
    os.makedirs(f"{full_output_dir}_final_positions", exist_ok = True) # HACK
    gammas = np.linspace(g_start, g_end, g_step)
    base_config_dict = asdict(config)
    print("Beginning to parralel quench")
    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        futures = []
        for gamma in gammas:
            futures.append(executor.submit(run_single_quench, gamma, base_config_dict, loadfile, output_dir))
        for f in futures:
            f.result()

    print("All quench simulations completed.")


def generate_density_map_images_from_quench_folder(traj_folder: str, output_dir: str, base_config: SimulationConfig, num_workers = None, square = False, extent = None, extension = "png"):
    print(f"Beginning to parralel process trajectory files")
    os.makedirs(os.path.join(base_config.output_path, output_dir), exist_ok = True)
    
    output_folder = base_config.output_path
    traj_files = sorted(
        glob.glob(os.path.join(output_folder,traj_folder, "*.json")),
        key=lambda f: float(re.search(r"_(\d+\.\d+)_traj", os.path.basename(f)).group(1)),
        reverse=True
    )

    if extent == None:
        x_min, x_max, y_min, y_max = find_xy_bounds(traj_files[0])
    
    else:
        x_min = extent[0]
        x_max = extent[1]
        y_min = extent[2]
        y_max = extent[3]
    
    base_config_dict = asdict(base_config)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for index, traj_file in enumerate(traj_files):
            futures.append(executor.submit(
                process_single_density_map,
                traj_file,
                output_dir,
                base_config_dict,
                x_min,
                x_max,
                y_min,
                y_max,
                index,
                square,
                extension
            ))

        # Optionally block until all are done
        for f in futures:
            f.result()
    print("Done!")
    
def generate_rolling_average_density_map_images_from_quench_folder(
    traj_folder: str,
    output_dir: str,
    base_config: SimulationConfig,
    time_per_image: float = 100,
    overlapping_time: float = 0,
    full_histogram: bool = True,
    animate: bool = False,
    num_workers: int = None,
    square: bool = False,
    extent = None,
    extension: str = "png"
):

    print("🔁 Beginning parallel rolling-average density map generation")

    output_folder = base_config.output_path
    traj_files = sorted(
        glob.glob(os.path.join(output_folder, traj_folder, "*.json")),
        key=lambda f: float(re.search(r"_(\d+\.\d+)_traj", os.path.basename(f)).group(1)),
        reverse=True
    )

    
    if extent == None:
        x_min, x_max, y_min, y_max = find_xy_bounds(traj_files[0])
    
    else:
        x_min = extent[0]
        x_max = extent[1]
        y_min = extent[2]
        y_max = extent[3]
    
    base_config_dict = asdict(base_config)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for index, traj_file in enumerate(traj_files):
            traj_basename = os.path.splitext(os.path.basename(traj_file))[0]
            traj_output_dir = os.path.join(output_dir, traj_basename)

            futures.append(
                executor.submit(
                    process_single_rolling_average_density_map,
                    traj_file,
                    traj_output_dir,
                    base_config_dict,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    index,
                    square,
                    extension,
                    time_per_image,
                    overlapping_time,
                    full_histogram,
                    animate
                )
            )

        for f in futures:
            f.result()  # To catch exceptions

    print("✅ All rolling-average density maps generated.")

    

def generate_temperature_plots_from_quench_folder(
    traj_folder,
    output_dir,
    base_config,
    grainyness = 100,
    num_workers = None,
    extension = "png"
):
    print("🔁 Beginning parallel temperature plot generation")
    os.makedirs(os.path.join(base_config.output_path, output_dir), exist_ok=True)

    traj_files = sorted(
        glob.glob(os.path.join(base_config.output_path, traj_folder, "*.json")),
        key = lambda f: float(re.search(r"_(\d+\.\d+)_traj", os.path.basename(f)).group(1)),
        reverse=True
    )

    base_config_dict = asdict(base_config)

    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        futures = [
            executor.submit(
                process_single_temperature_plot,
                traj_file,
                output_dir,
                base_config_dict,
                grainyness,
                extension
            )
            for traj_file in traj_files
        ]
        for f in futures:
            f.result()  # To catch exceptions if any

    print("✅ All temperature plots generated.")