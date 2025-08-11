#Written 2025-07-25 by Jack Beda (jack.beda.ca).
###############################################################

'''
This is the real meat and potatos of the simulation. It is here that you can initialize simulations, run them, etc. While most of the time you are interested in running many simulations in parralel (see parralel.py), you can also run and visualize individual simulations and trajectories. this contains:

    SimulationConfig
        # This class produces an object we usually call 'config' or 'base_config' that is basically a dictionary to store
        # all of the simulation parameters. If you want to save the details of the simulation, running config.save_shortform()
        # will save you a nice text file with the details of the simulation. Nearly everything will require you to pass it a
        # config in some form or another.
        
    SimulationState
        # This handles all of the parts of the simulation that change with time (positions/energies/etc). It also includes some
        # functions that let you minimize the energy of the system (minimize_forces and minimize_energy). These are useful
        # when finding stable points of the system as starting points for quenches.
        
    SimulationRunner
        # Now on to running the simulation. You'll take a SimulationConfig object and a SimulationState object, pass that to
        # SimulationRunner and it'll update SimulationState as it runs the simulation.
        
    SimulationIO
        # This just includes some helpful functions for loading data like trajectories or positions into SimulationState, or
        # or saving the current state to file.
        
        
    SimulationVisualizer
        # To view things like trajectories, density maps, and positions, just make a visualizer = SimulationVisualizer() object
        # and then call something like visualizer.plot_positions(state) where you pass the state to the system.
    
    AnimationMaker
        #
'''


import numpy as np
import os
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, List
import natural_units as units
import utility_functions as utility
from laser import Laser
from matplotlib.patches import Ellipse
import imageio.v2 as imageio
from scipy.optimize import minimize, root
from time import time
import glob
import re  # Needed to extract gamma values
from concurrent.futures import ProcessPoolExecutor
import subprocess
from scipy.optimize import minimize, root

@dataclass
class SimulationConfig:
    N: int
    w: float
    g: float
    m: float
    T_mK: float
    dt: float
    num_steps: int
    damping: bool = False
    damping_parameter: float = 1.0
    langevin_temperature: bool = False
    lasers: Optional[List[Laser]] = None
    input_path: str = None
    output_path: str = None
        
    
    def get_shortform(self) -> str:
        """Return a short string summary of the simulation configuration."""
        summary = (
            f"N = {self.N},\n"
            f"w = {self.w},\n"
            f"g = {self.g},\n"
            f"m = {self.m} amu,\n"
            f"T = {self.T_mK} mK,\n"
            f"dt = {self.dt} μs,\n"
            f"time = {self.dt * self.num_steps} μs,\n"
            f"num_steps = {self.num_steps},\n"
            f"damping={self.damping},\n"
            f"damping_parameter = {self.damping_parameter},\n"
            f"langevin_temperature = {self.langevin_temperature},\n"
            f"number of lasers={len(self.lasers) if self.lasers else 0}\n"
        )
        
        for i,laser in enumerate(self.lasers):
            summary += f"laser {i}:\n"
            summary += f"    wavelength (λ) = {laser.wavelength * 1000} nm\n"
            summary += f"    direction unit vector = {laser.direction}\n"
            summary += f"    saturation (s) = {laser.saturation}\n"
            summary += f"    detuning (Δ) = {laser.detuning} MHz\n"
            summary += f"    Gamma (Γ) = {laser.Gamma} MHz\n"
        
        return summary

    def save_shortform(self, filename: str = "simulation_details.txt") -> None:
        """Save the shortform summary to a text file."""
        
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok = True)
            filename = os.path.join(self.output_path, filename)
            
        with open(filename, 'w') as f:
            f.write(self.get_shortform())

class SimulationState:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.N = config.N
        self.positions = np.zeros((self.N, 2))
        self.velocities = np.zeros((self.N, 2))
        self.trajectory = np.zeros((self.config.num_steps, self.N, 2))
        self.velocity_trajectory = np.zeros_like(self.trajectory)
        self.temperatures = np.zeros(self.config.num_steps // 100)
        self.potential_energies = np.zeros_like(self.temperatures)
        self.kinetic_energies = np.zeros_like(self.temperatures)
        self.total_energies = np.zeros_like(self.temperatures)
        self.times = np.zeros_like(self.temperatures)
        self.initialized = False
        self.initial_positions = None

    def initialize_positions(self, method='random'):
        if method == 'circle':
            angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
            radius = utility.base_radius(self.N, self.config.m, self.config.w, units.k)
            self.positions = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        elif method == 'random':
            radius = utility.base_radius(self.N, self.config.m, self.config.w, units.k)
            positions = []
            while len(positions) < self.N:
                points = np.random.uniform(-radius, radius, size=(self.N * 2, 2))
                distances = np.linalg.norm(points, axis=1)
                inside = points[distances <= radius]
                positions.extend(inside.tolist())
            self.positions = np.array(positions[:self.N])
        
        
        else:
            raise ValueError(f"No initialization method known by name '{method}'")

        self.initial_positions = self.positions.copy()
        self.initialized = True
        

    def reset(self):
        self.positions = self.initial_positions.copy()
        self.velocities = np.zeros_like(self.velocities)
        self.trajectory = np.zeros_like(self.trajectory)
        self.velocity_trajectory = np.zeros_like(self.velocity_trajectory)
        
    def temperature_from_velocities(self):
        return 0.5 * self.config.m * np.sum(self.velocities ** 2) / units.kB
    
    def minimize_energy(self):
        """
        Minimize the potential energy of the system and update self.positions.
        Uses scipy.optimize.minimize with Powell method.
        """
        from scipy.optimize import minimize

        def potential_energy(flat_pos):
            positions = flat_pos.reshape((self.N, 2))
            x, y = positions[:, 0], positions[:, 1]
            trap_energy = 0.5 * self.config.m * self.config.w**2 * np.sum(x**2 + (self.config.g * y)**2)
            coulomb_term = 0.0
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    r = np.linalg.norm(positions[i] - positions[j]) + 1e-8  # Avoid div by 0
                    coulomb_term += 1 / r
            return trap_energy + units.k * coulomb_term

        print(f"🔍 Minimizing potential energy for N = {self.N} ions...")
        initial_flat = self.positions.flatten()
        result = minimize(potential_energy, initial_flat, method='Powell')

        if not result.success:
            raise RuntimeError(f"Minimization failed: {result.message}")

        self.positions = result.x.reshape((self.N, 2))
        self.initial_positions = self.positions.copy()
        self.initialized = True
        print(f"✅ Minimization complete. Final energy: {result.fun:.6f}")
        
    def minimize_forces(self):
        """
        Find positions where net forces on all particles vanish.
        Solves F = 0 using scipy.optimize.root and updates self.positions.
        """

        def force_equations(flat_pos):
            positions = flat_pos.reshape((self.N, 2))
            forces = np.zeros_like(positions)

            wx2 = self.config.w ** 2
            wy2 = (self.config.w * self.config.g) ** 2
            m = self.config.m

            # Harmonic trap
            forces -= m * positions * np.array([wx2, wy2])

            # Coulomb forces
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        continue
                    r_vec = positions[i] - positions[j]
                    r = np.linalg.norm(r_vec) + 1e-8  # avoid div by zero
                    forces[i] += units.k * r_vec / r**3

            return forces.flatten()

        print(f"🔍 Solving for force balance (∑F = 0) for N = {self.N} ions...")
        initial_flat = self.positions.flatten()
        result = root(force_equations, initial_flat, method='hybr')

        if not result.success:
            raise RuntimeError(f"Force minimization failed: {result.message}")

        self.positions = result.x.reshape((self.N, 2))
        self.initial_positions = self.positions.copy()
        self.initialized = True
        print("✅ Force minimization complete. All net forces are near zero.")



class SimulationRunner:
    def __init__(self, config: SimulationConfig, state: SimulationState):
        self.config = config
        self.state = state
        self.N = config.N
        self.m = config.m
        self.dt = config.dt
        self.T = config.T_mK / 0.120272422607
        self.w = config.w
        self.g = config.g
        self.wx2 = self.w**2
        self.wy2 = (self.w * self.g)**2

    def compute_forces(self):
        pos = self.state.positions
        vel = self.state.velocities
        N = self.N

        pairwise_forces = compute_coulomb_forces_numpy(pos)
        harmonic_forces = -self.m * pos * np.array([self.wx2, self.wy2])
        cooling_forces = np.zeros_like(vel)
        if self.config.lasers:
            for laser in self.config.lasers:
                cooling_forces += laser.get_force(vel)
        noise_forces = 0.
        if self.config.langevin_temperature and self.T > 0:
            sigma = np.sqrt(2 * units.kB * self.config.damping_parameter * self.T / self.dt)
            noise_forces = sigma * np.random.normal(size=(N, 2))
        damping_forces = -self.config.damping_parameter * vel if self.config.damping else 0.
        return pairwise_forces + harmonic_forces + cooling_forces + noise_forces + damping_forces

    def run(self, verbose = False, grainyness = 100):
        print(f"Running with isotropy (γ) = {self.config.g:.4f} for {self.config.num_steps * self.config.dt} μs")
        if not self.state.initialized:
            self.state.initialize_positions()
            print("Initializing random positions")
        self.state.velocities = self.get_thermal_velocities()

        for step in range(self.config.num_steps):
            forces = self.compute_forces()
            self.state.trajectory[step] = self.state.positions
            self.state.velocity_trajectory[step] = self.state.velocities
            acc = forces / self.m
            new_pos = self.state.positions + self.dt * self.state.velocities + 0.5 * acc * self.dt**2
            self.state.velocities += acc * self.dt
            self.state.positions = new_pos

            if verbose and step % grainyness == 0:
                i = step // grainyness
                ke = self.get_kinetic_energy()
                pe = self.get_potential_energy()
                if i < len(self.state.temperatures):
                    self.state.temperatures[i] = self.get_temperature(ke)
                    self.state.kinetic_energies[i] = ke
                    self.state.potential_energies[i] = pe
                    self.state.total_energies[i] = ke + pe
                    self.state.times[i] = step * self.dt

    def get_thermal_velocities(self):
        v = np.random.randn(self.N, 2)
        desired_ke = units.kB * self.N * self.T
        current_ke = 0.5 * self.m * np.sum(v ** 2)
        v *= np.sqrt(desired_ke / current_ke)
        return v

    def get_kinetic_energy(self):
        return 0.5 * self.m * np.sum(self.state.velocities**2)

    def get_potential_energy(self):
        x, y = self.state.positions[:, 0], self.state.positions[:, 1]
        term1 = 0.5 * self.m * self.w**2 * np.sum(x**2 + (self.g * y)**2)
        term2 = 0.
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = np.linalg.norm(self.state.positions[i] - self.state.positions[j])
                term2 += 1 / r
        return term1 + units.k * term2

    def get_temperature(self, kinetic_energy):
        return kinetic_energy / (units.kB * self.N)
    
    
class SimulationIO:
    def __init__(self, filename = None,
                 output_path = None,
                 input_path = None):
        self.output_path = output_path
        self.input_path = input_path
        self.filename = filename

    def _resolve_filename(self, filename = None) -> str:
        fname = filename if filename is not None else self.filename
        if not fname:
            raise ValueError("No filename provided and self.filename is not set.")
        return fname

    def _full_output_path(self, filename = None) -> str:
        """Join output_path and filename; if no output_path, return filename."""
        fname = self._resolve_filename(filename)
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            return os.path.join(self.output_path, fname)
        return fname

    def _full_input_path(self, filename = None) -> str:
        """Join input_path and filename; if no input_path, return filename."""
        fname = self._resolve_filename(filename)
        if self.input_path:
            return os.path.join(self.input_path, fname)
        return fname

    # ---- I/O methods ----

    def save_positions(self, positions: np.ndarray, filename = None) -> None:
        path = self._full_output_path(filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(positions.tolist(), f)

    def save_trajectory(self, trajectory: np.ndarray, filename = None) -> None:
        path = self._full_output_path(filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(trajectory.tolist(), f)

    def load_positions(self, filename = None) -> np.ndarray:
        path = self._full_input_path(filename)
        with open(path, 'r', encoding='utf-8') as f:
            return np.array(json.load(f))

    def save_config(self, config, filename = None) -> None:
        path = self._full_output_path(filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f)

    def load_config(self, filename = None):
        path = self._full_input_path(filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return SimulationConfig(**data)
    
###########################################################################################




# Visualization utilities for SimulationState
class SimulationVisualizer:
    
    def __init__(self, save = False, filename = None, extension = "png", output_path = None):
        self.save = save
        self.filename = filename
        self.extension = extension
        self.output_path = output_path
        
    def _full_path(self, filename):
        """Join output_path and filename if output_path is set; create dirs if needed."""
        path = os.path.join(self.output_path, filename) if self.output_path else filename
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok = True)
        return path

    def _pick_name(self, default_stub):
        """
        Choose the filename to save:
        - use self.filename if set,
        - otherwise use f'{default_stub}.{self.extension}'
        """
        base = self.filename if self.filename is not None else f"{default_stub}.{self.extension}"
        
        return self._full_path(base)

        
    def plot_positions(self, state: SimulationState, square=True):
        x = state.positions[:, 0]
        y = state.positions[:, 1]

        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, color='blue', s=100)
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi, yi, f"{i+1}", fontsize=12, ha='right', color='red')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.xlabel('x-coordinate (μm)')
        plt.ylabel('y-coordinate (μm)')
        plt.title(f'Particle Positions for N = {state.N}')
        plt.grid(True)
        if square:
            plt.gca().set_aspect('equal', adjustable='box')
        
        if self.save:
            plt.savefig(self._pick_name(f"{state.N}_positions_plot"))
            
        plt.show()

    def plot_trajectory(self, state: SimulationState, square=True):
        plt.figure(figsize=(8, 8))
        for i in range(state.N):
            plt.plot(state.trajectory[:, i, 0], state.trajectory[:, i, 1])
            plt.scatter(state.trajectory[0, i, 0], state.trajectory[0, i, 1], color='red', label=f'Start {i+1}' if i == 0 else "")
            plt.scatter(state.trajectory[-1, i, 0], state.trajectory[-1, i, 1], color='blue', label=f'End {i+1}' if i == 0 else "")
        plt.title(f"Trajectories of {state.N} Particles")
        plt.xlabel("x-coordinate (μm)")
        plt.ylabel("y-coordinate (μm)")
        plt.legend()
        plt.grid()
        if square:
            plt.gca().set_aspect('equal', adjustable='box')
        
        if self.save:
            plt.savefig(self._pick_name(f"{state.N}_trajectory_plot"))
            
        plt.show()

    def plot_density_map(self, state: SimulationState, bins=100, cmap='Greys'):
        positions = state.trajectory.reshape(-1, 2)
        x = positions[:, 0]
        y = positions[:, 1]

        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
        hist = hist.T
        hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(hist_norm, origin='lower', cmap=cmap, extent=extent, interpolation='nearest')
        ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_title('Ion Density Map')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, label='Normalized density')
        
        if self.save:
            plt.savefig(self._pick_name(f"{state.N}_density_plot"))

        else:
            plt.show()
               
    def plot_energy(self, state: SimulationState):
        t = state.times
        plt.figure(figsize=(10, 6))
        plt.plot(t, state.kinetic_energies, label="Kinetic Energy")
        plt.plot(t, state.potential_energies, label="Potential Energy")
        plt.plot(t, state.total_energies, label="Total Energy", linestyle='--')
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Energies Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if self.save:
            plt.savefig(self._pick_name(f"{state.N}_energy_plot"))
            
        plt.show()
        
    def plot_temperature(self, state: SimulationState):
        t = state.times
        temps_mK = units.theta_to_mK(state.temperatures)
        plt.figure(figsize=(10, 4))
        plt.plot(t, temps_mK, label = "Temperature")
        plt.xlabel("Time (μs)")
        plt.ylabel("Temperature (mK)")
        plt.title("System Temperature Over Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        if self.save:
            plt.savefig(self._pick_name(f"{state.N}_temperature_plot"))
            
        plt.show()
        
class AnimationMaker:
#     @staticmethod
#     def generate_density_map_images_from_quench_folder(traj_folder: str, output_dir: str, base_config: SimulationConfig):
#         print(f"Processing trajectory files")
#         os.makedirs(output_dir, exist_ok=True)

#         # Sort trajectories by gamma (extracted from filenames), descending
#         traj_files = sorted(
#             glob.glob(os.path.join(traj_folder, "*.json")),
#             key=lambda f: float(re.search(r"_(\d+\.\d+)_traj", os.path.basename(f)).group(1)),
#             reverse=True
#         )

#         # Determine plot bounds from the trajectory with largest gamma
#         with open(traj_files[0], 'r') as f:
#             traj = np.array(json.load(f)).reshape(-1, 2)
#         x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
#         y_min, y_max = traj[:, 1].min(), traj[:, 1].max()

#         for i, traj_file in enumerate(traj_files):
#             # Extract gamma from filename
#             match = re.search(r"_(\d+\.\d+)_traj", os.path.basename(traj_file))
#             gamma = float(match.group(1)) if match else 0.0
#             filename = os.path.join(output_dir, f"frame_{i:03d}_gamma_{gamma:.6f}.png")

#             if os.path.exists(filename):
#                 print(f"Skipping frame {i:03d} (γ = {gamma:.6f}), image already exists.")
#                 continue

#             print(f"Processing {traj_file}")
#             try:
#                 with open(traj_file, 'r') as f:
#                     traj = np.array(json.load(f))
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error in file: {traj_file}")
#                 print(e)
#                 continue
#             except FileNotFoundError as e:
#                 print(f"File not found: {traj_file}")
#                 print(e)
#                 continue

#             # Clone base config and override gamma + steps
#             config = SimulationConfig(**{**asdict(base_config), 'g': gamma, 'num_steps': traj.shape[0]})
#             state = SimulationState(config)
#             state.trajectory = traj
#             state.positions = traj[-1]
#             state.initialized = True

#             # Generate density map
#             positions = state.trajectory.reshape(-1, 2)
#             x, y = positions[:, 0], positions[:, 1]
#             hist, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[x_min, x_max], [y_min, y_max]])
#             hist = hist.T
#             hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
#             extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#             fig, ax = plt.subplots(figsize=(6, 6))
#             im = ax.imshow(hist_norm, origin='lower', cmap='Greys', extent=extent, interpolation='nearest')
#             ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)
#             ax.set_xlabel('x position')
#             ax.set_ylabel('y position')
#             ax.set_title(f'Ion Density Map: γ = {gamma:.6f}')
#             ax.set_aspect('equal')
#             fig.colorbar(im, ax=ax, label='Normalized density')
#             plt.savefig(filename)
#             plt.close(fig)
            
#     def generate_temperature_plots_from_quench_folder(traj_folder, output_dir, base_config, grainyness = 100):

#         def estimate_temperature(trajectory, dt, m, grainyness):
#             n_steps, N, _ = trajectory.shape
#             n_windows = n_steps // grainyness - 1
#             temperatures = []
#             times = []
#             for i in range(n_windows):
#                 idx1 = i * grainyness
#                 idx2 = i * grainyness + 1
#                 delta_r = trajectory[idx2] - trajectory[idx1]
#                 v_avg = delta_r / dt
#                 kinetic_energy = 0.5 * m * np.sum(v_avg**2)
#                 T = kinetic_energy / (N * units.kB)
#                 temperatures.append(T)
#                 times.append(idx1 * dt)
#             return np.array(times), np.array(temperatures)

#         os.makedirs(output_dir, exist_ok = True)

#         traj_files = sorted(
#             glob.glob(os.path.join(traj_folder, "*.json")),
#             key=lambda f: float(re.search(r"_(\d+\.\d+)_traj", os.path.basename(f)).group(1)),
#             reverse=True
#         )

#         for traj_file in traj_files:
#             match = re.search(r"_(\d+\.\d+)_traj", os.path.basename(traj_file))
#             gamma = float(match.group(1)) if match else 0.0
#             outfile = os.path.join(output_dir, f"temperature_plot_{gamma:.6f}.png")

#             try:
#                 with open(traj_file, 'r') as f:
#                     traj = np.array(json.load(f))
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error in file: {traj_file}")
#                 print(e)
#                 continue
#             except FileNotFoundError as e:
#                 print(f"File not found: {traj_file}")
#                 print(e)
#                 continue

#             if traj.ndim != 3 or traj.shape[1:] != (base_config.N, 2):
#                 print(f"Skipping {traj_file}, unexpected shape {traj.shape}")
#                 continue

#             times, temps = estimate_temperature(traj, base_config.dt, base_config.m, grainyness)
#             temps_mK = units.theta_to_mK(temps) 
#             plt.figure(figsize=(8, 4))
#             plt.plot(times, temps_mK, label = "Temperature (mK)")  # Convert to mK
#             plt.scatter(times[0], temps_mK[0], label = f"T0 = {temps_mK[0]:.3f} mK")
#             plt.legend()
#             plt.xlabel("Time (μs)")
#             plt.ylabel("Temperature (mK)")
#             plt.title(f"Estimated Temperature vs Time (γ = {gamma:.4f})")
#             plt.grid(True)
#             plt.tight_layout()
#             plt.savefig(outfile)
#             plt.close()

#         print("✅ All temperature plots generated.")


    @staticmethod
    def make_gif_or_mp4_from_images(image_folder: str, output_file: str, fps: int = 4, reverse: bool = True):
        images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
        if not images:
            raise FileNotFoundError(f"No PNG files found in {image_folder}. Maybe you have pdf images instead? This function cannot handle pdf images as input")

        if reverse:
            images = images[::-1]

        ext = os.path.splitext(output_file)[-1].lower()

        if ext == ".gif":
            frames = [imageio.imread(img) for img in images]
            imageio.mimsave(output_file, frames, duration=1 / fps)

        elif ext == ".mp4":
            frames = [imageio.imread(img) for img in images]
            try:
                imageio.mimsave(output_file, frames, fps=fps, codec='libx264')
                print(f"MP4 saved via imageio: {output_file}")
            except Exception as e:
                print(f"[Warning] imageio failed to save MP4: {e}")
                print("[Fallback] Trying subprocess + ffmpeg...")
                # Fallback to FFmpeg
                output_dir = os.path.dirname(output_file)
                output_name = os.path.basename(output_file)
                os.makedirs(output_dir, exist_ok=True)
                cmd = [
                    "/exports/eddie/scratch/s2142953/miniconda3/bin/ffmpeg",
                    "-y",
                    "-framerate", str(fps),
                    "-pattern_type", "glob",
                    "-i", os.path.join(image_folder, "*.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "--",
                    output_name
                ]
                subprocess.run(cmd, cwd=output_dir, check=True)
                print(f"MP4 saved via ffmpeg fallback: {output_file}")

        else:
            raise ValueError("Output must be .gif or .mp4")
            
            
# Helper force computation function

def compute_coulomb_forces_numpy(pos):
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(delta**2, axis=-1) + 1e-12
    np.fill_diagonal(r2, np.inf)
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    forces = np.sum(delta * inv_r3[:, :, np.newaxis], axis=1)
    return units.k * forces
