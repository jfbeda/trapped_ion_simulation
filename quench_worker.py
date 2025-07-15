
import numpy as np
import os
import json
from dataclasses import asdict
from simulation_module import SimulationConfig, SimulationState, SimulationRunner, SimulationIO  # Adjust import if needed

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
