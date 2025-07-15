
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

@dataclass
class SimulationConfig:
    N: int
    w: float
    g: float
    m: float
    T: float
    dt: float
    num_steps: int
    damping: bool = False
    damping_parameter: float = 1.0
    langevin_temperature: bool = False
    lasers: Optional[List[Laser]] = None

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
        else:
            radius = utility.base_radius(self.N, self.config.m, self.config.w, units.k)
            positions = []
            while len(positions) < self.N:
                points = np.random.uniform(-radius, radius, size=(self.N * 2, 2))
                distances = np.linalg.norm(points, axis=1)
                inside = points[distances <= radius]
                positions.extend(inside.tolist())
            self.positions = np.array(positions[:self.N])
        self.initial_positions = self.positions.copy()
        self.initialized = True

    def reset(self):
        self.positions = self.initial_positions.copy()
        self.velocities = np.zeros_like(self.velocities)
        self.trajectory = np.zeros_like(self.trajectory)
        self.velocity_trajectory = np.zeros_like(self.velocity_trajectory)


class SimulationRunner:
    def __init__(self, config: SimulationConfig, state: SimulationState):
        self.config = config
        self.state = state
        self.N = config.N
        self.m = config.m
        self.dt = config.dt
        self.T = config.T / 0.120272422607
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
    @staticmethod
    def save_trajectory(trajectory, filename):
        with open(filename, 'w') as f:
            json.dump(trajectory.tolist(), f)

    @staticmethod
    def load_positions(filename):
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    @staticmethod
    def save_config(config: SimulationConfig, filename):
        with open(filename, 'w') as f:
            json.dump(asdict(config), f)

    @staticmethod
    def load_config(filename) -> SimulationConfig:
        with open(filename, 'r') as f:
            data = json.load(f)
        return SimulationConfig(**data)
    
###########################################################################################


    
def run_quench_series_parallel(config: SimulationConfig, loadfile: str, output_dir: str,
                               g_start: float, g_end: float, g_step: int, num_workers: int = 4):
    os.makedirs(output_dir, exist_ok=True)
    gammas = np.linspace(g_start, g_end, g_step)
    base_config_dict = asdict(config)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for gamma in gammas:
            futures.append(executor.submit(run_single_quench, gamma, base_config_dict, loadfile, output_dir))
        for f in futures:
            f.result()  # wait and raise errors if any

    print("All quench simulations completed.")


# class QuenchWorkflow:
#     def __init__(self, config: SimulationConfig, loadfile: str, output_dir: str):
#         self.base_config = config
#         self.loadfile = loadfile
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
        

#     def run_quench_series(self, g_start, g_end, g_step):
#         gammas = np.linspace(g_start, g_end, g_step)

#         for gamma in gammas:
#             out_file = f"{self.output_dir}/{self.base_config.N}_{gamma:.14f}_traj_{self.base_config.num_steps}_steps.json"
#             if os.path.exists(out_file):
#                 print(f"Skipping gamma = {gamma:.8f}, file exists.")
#                 continue
                
#             print(f"Quenching to isotropy (γ) = {gamma:.8f}")

#             # Create new config and state
#             config = SimulationConfig(**{**asdict(self.base_config), 'g': gamma, 'w': self.base_config.w})
#             state = SimulationState(config)

#             # Load same initial structure every time
#             state.positions = SimulationIO.load_positions(self.loadfile)
#             state.initial_positions = state.positions.copy()
#             state.initialized = True
#             state.reset()  # randomize velocities

#             # === Quench trap frequency to conserve potential energy ===
#             x_vars = state.positions[:, 0]
#             y_vars = state.positions[:, 1]

#             initial_g = self.base_config.g
#             initial_w = self.base_config.w
#             m = self.base_config.m

#             initial_PE = 0.5 * m * initial_w**2 * np.sum(x_vars**2 + (initial_g * y_vars)**2)
#             new_PE = 0.5 * m * initial_w**2 * np.sum(x_vars**2 + (gamma * y_vars)**2)

#             new_w = initial_w * np.sqrt(initial_PE / new_PE)
#             print(f"Adjusted w for energy conservation: {initial_w:.4f} → {new_w:.4f}")
#             state.w = new_w
#             state.g = gamma
#             state.config.w = new_w
#             state.config.g = gamma

#             # Run and save
#             runner = SimulationRunner(state.config, state)
#             runner.run()
#             SimulationIO.save_trajectory(state.trajectory, out_file)
        
#         print("Done!")


# Visualization utilities for SimulationState
class SimulationVisualizer:
    @staticmethod
    def plot_positions(state: SimulationState, square=True, save=False, filename=None):
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
        if save:
            if filename is None:
                filename = f"{state.N}_positions_plot.png"
            plt.savefig(filename)
        plt.show()

    @staticmethod
    def plot_trajectory(state: SimulationState, square=True):
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
        plt.show()

    @staticmethod
    def plot_density_map(state: SimulationState, bins=100, cmap='Greys', save=False, filename=None):
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
        if save:
            if filename is None:
                filename = "density_map.png"
            plt.savefig(filename)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_energy(state: SimulationState):
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
        plt.show()
        
    @staticmethod
    def plot_temperature(state: SimulationState):
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
        plt.show()

        
class AnimationMaker:
    @staticmethod
    def generate_density_map_images_from_quench_folder(traj_folder: str, output_dir: str, base_config: SimulationConfig):
        print(f"Processing trajectory files")
        os.makedirs(output_dir, exist_ok=True)

        # Sort trajectories by gamma (extracted from filenames), descending
        traj_files = sorted(
            glob.glob(os.path.join(traj_folder, "*.json")),
            key=lambda f: float(re.search(r"_(\d+\.\d+)_traj", os.path.basename(f)).group(1)),
            reverse=True
        )

        # Determine plot bounds from the trajectory with largest gamma
        with open(traj_files[0], 'r') as f:
            traj = np.array(json.load(f)).reshape(-1, 2)
        x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
        y_min, y_max = traj[:, 1].min(), traj[:, 1].max()

        for i, traj_file in enumerate(traj_files):
            # Extract gamma from filename
            match = re.search(r"_(\d+\.\d+)_traj", os.path.basename(traj_file))
            gamma = float(match.group(1)) if match else 0.0
            filename = os.path.join(output_dir, f"frame_{i:03d}_gamma_{gamma:.6f}.png")

            if os.path.exists(filename):
                print(f"Skipping frame {i:03d} (γ = {gamma:.6f}), image already exists.")
                continue

            print(f"Processing {traj_file}")
            with open(traj_file, 'r') as f:
                traj = np.array(json.load(f))

            # Clone base config and override gamma + steps
            config = SimulationConfig(**{**asdict(base_config), 'g': gamma, 'num_steps': traj.shape[0]})
            state = SimulationState(config)
            state.trajectory = traj
            state.positions = traj[-1]
            state.initialized = True

            # Generate density map
            positions = state.trajectory.reshape(-1, 2)
            x, y = positions[:, 0], positions[:, 1]
            hist, xedges, yedges = np.histogram2d(x, y, bins=100, range=[[x_min, x_max], [y_min, y_max]])
            hist = hist.T
            hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(hist_norm, origin='lower', cmap='Greys', extent=extent, interpolation='nearest')
            ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')
            ax.set_title(f'Ion Density Map: γ = {gamma:.6f}')
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax, label='Normalized density')
            plt.savefig(filename)
            plt.close(fig)
            
    def generate_temperature_plots_from_quench_folder(traj_folder, output_dir, base_config, grainyness = 100):

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

        os.makedirs(output_dir, exist_ok = True)

        traj_files = sorted(
            glob.glob(os.path.join(traj_folder, "*.json")),
            key=lambda f: float(re.search(r"_(\d+\.\d+)_traj", os.path.basename(f)).group(1)),
            reverse=True
        )

        for traj_file in traj_files:
            match = re.search(r"_(\d+\.\d+)_traj", os.path.basename(traj_file))
            gamma = float(match.group(1)) if match else 0.0
            outfile = os.path.join(output_dir, f"temperature_plot_{gamma:.6f}.png")

            with open(traj_file, 'r') as f:
                traj = np.array(json.load(f))

            if traj.ndim != 3 or traj.shape[1:] != (base_config.N, 2):
                print(f"Skipping {traj_file}, unexpected shape {traj.shape}")
                continue

            times, temps = estimate_temperature(traj, base_config.dt, base_config.m, grainyness)
            temps_mK = units.theta_to_mK(temps) 
            plt.figure(figsize=(8, 4))
            plt.plot(times, temps_mK, label = "Temperature (mK)")  # Convert to mK
            plt.scatter(times[0], temps_mK[0], label = f"T0 = {temps_mK[0]:.3f} mK")
            plt.legend()
            plt.xlabel("Time (μs)")
            plt.ylabel("Temperature (mK)")
            plt.title(f"Estimated Temperature vs Time (γ = {gamma:.4f})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(outfile)
            plt.close()

        print("✅ All temperature plots generated.")


    @staticmethod
    def make_gif_or_mp4_from_images(image_folder: str, output_file: str, fps: int = 4):
        images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
        frames = [imageio.imread(img) for img in images]
        ext = os.path.splitext(output_file)[-1].lower()
        if ext == ".gif":
            imageio.mimsave(output_file, frames, duration=1 / fps)
        elif ext == ".mp4":
            imageio.mimsave(output_file, frames, fps=fps, codec='libx264')
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
