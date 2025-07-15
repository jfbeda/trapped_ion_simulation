
# Written 2025-07-14
import numpy as np
import math

import numba
import os

from scipy.optimize import minimize
from scipy.optimize import root
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import imageio.v2 as imageio

from time import time
from datetime import datetime
import json

import natural_units as units
import utility_functions as utility
from laser import Laser


class Simulation:
    
    def __init__(self, N, w, g, m, T, dt, num_steps, damping, damping_parameter, langevin_temperature = False, lasers = None):
        self.N = N                                     # Number of particles
        self.w = w                                     # Trap frequency (MHz)
        self.g = g                                     # Trap isotropy (0 < g ≤ 1), dimensionless
        self.m = m                                     # Ion mass (amu)
        self.half_m = self.m/2                         # Half ion mass (amu)
        self.T = T/0.120272422607                      # Temperature (1 θ = κ/k_B ≈ 0.12 mK)
        self.dt = dt                                   # Time step (μs)
        self.num_steps = num_steps                     # Number of simulation steps
        self.simulation_time = num_steps * self.dt     # Total simulation time (μs)
        self.grainyness = 100                          # Number of steps to take data on verbose runs
        
        num_recordings = self.num_steps // self.grainyness
        self.temperatures = np.zeros(num_recordings)
        self.potential_energies = np.zeros(num_recordings)
        self.kinetic_energies = np.zeros(num_recordings)
        self.total_energies = np.zeros(num_recordings)
        self.times = np.zeros(num_recordings)
        
        self.damping = damping                         # Whether or not damping is enabled (bool)
        self.damping_parameter = damping_parameter     # Damping parameter (amu/μs)
        self.langevin_temperature = langevin_temperature # Whether or not langevin thermostat force enabled (bool)
        
        self.wx = w                                    # Trap frequency in x-direction (MHz)
        self.wx2 = self.wx**2                          # Square trap frequency in x-direction (MHz^2)
        self.wy = w * g                                # Trap frequency in y-direction (MHz)
        self.wy2 = self.wy**2                          # Square trap frequency in y-direction (MHz^2)
        self.initialized = False                       # Flag for simulation initialization
        self.initial_positions = np.zeros((self.N, 2)) # Initial positions (μm)
        self.positions = self.initial_positions        # Current positions (μm)
        self.velocities = np.zeros((self.N, 2))        # Velocities (μm/μs)
        self.trajectory = np.zeros((self.num_steps, self.N, 2)) # Particle trajectories (μm)
        self.velocity_trajectory = np.zeros((self.num_steps, self.N, 2)) # Particle velocities over time (μm/μs)
        
        self.lasers = lasers if (lasers is not None) else []
        
        
        self._attribute_map = {
            'positions': {
                'attr': 'positions',
                'default_filename': 'positions.json',
                'generate_filename': lambda self, count: f"{self.N}_{self.g}_positions_{count}.json"
            },
            'trajectory': {
                'attr': 'trajectory',
                'default_filename': 'trajectory.json',
                'generate_filename': lambda self, count: f"{self.N}_{self.g}_trajectory_{self.num_steps}_steps_{count}.json"
            },
            'temperature': {
                'attr': 'temperatures',
                'default_filename': 'temperature.json',
                'generate_filename': lambda self, count: f"{self.N}_{self.g}_temperature_{count}.json"
            }
        }

    
    def __str__(self):
        """
        Return a detailed string representation of the Simulation object.

        If the simulation is initialized, the output includes:
            - Number of particles (N)
            - Anisotropy parameter (g)
            - Temperature (in mK)
            - Particle mass (in atomic mass units)
            - Potential energy (total and per particle, in κ)
            - Kinetic energy (in κ)
            - Total energy (in κ)
            - RMS force (in amu·μm/μs²)
            - Stability metric (RMS displacement after a short run)
            - Average interparticle distance (in μm)

        If the simulation is not initialized, the output notes this status.

        Returns:
            str: Human-readable description of the simulation's current state and physical parameters.
        """
        
        potential_energy = self.get_energy()
        kinetic_energy = self.get_kinetic_energy()
        
        string = "Simulation object"
        
        if self.initialized:
            string += f"\nN = {self.N}, g = {self.g}, T = {units.theta_to_mK(self.T)} mK, m = {self.m} amu"
            string += f"\nPotential energy = {potential_energy} κ"
            string += f"\nPotential energy per particle = {potential_energy/self.N} κ"
            string += f"\nKinetic energy = {kinetic_energy} κ"
            string += f"\nTotal energy = {kinetic_energy + potential_energy} κ"
            string += f"\nRMS force: {np.linalg.norm(utility.fixed_point_equation(utility.positions_to_flattened_positions(self.positions), self.g, self.w, units.k, self.m))}"+ " amu μm / μs²"
            
            temp_T = self.T
            temp_num_steps = self.num_steps
            self.T = 1e-3 #Can be adjusted
            self.num_steps = 100
            
            string += f"\nStability: {self.check_stability()} (RMS displacement after running {self.num_steps} steps of simulation)"
            self.T = temp_T
            self.num_steps = temp_num_steps
            
            string += f"\nAverage interparticle distance = {self.get_average_interparticle_distance()} μm"
#            string += f"\nAverage distance from origin = {self.get_average_distance_from_center()}"
            
            string += f"\n ----------"
            string += f"\nRun print(simulation.units()) to view the unit conventions for this simulation"
        
        else:
            string += f"\nuninitialized"
        
        return string
    
    def reset(self):
        self.positions = self.initial_positions
        self.trajectory = np.zeros((self.num_steps, self.N, 2))
        self.velocities = np.zeros((self.N, 2))
            
    
    #############################################################
    #############################################################
    #############################################################
    # GET FUNCTIONS
    #############################################################
    #############################################################
    #############################################################
    
    def get_average_distance_from_center(self):
        """
        Calculate the average distance of all particles from the origin.

        Returns:
            float: Average distance from the origin (in μm).
        """
        return np.mean(np.linalg.norm(self.positions, axis=1))
    
    def get_average_interparticle_distance(self):
        """
        Computes the average nearest-neighbor distance between particles.

        Returns:
            float: Average distance to the nearest neighbor for each particle (in μm).
        """
        distances = np.linalg.norm(self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :], axis=2)
        distances[np.arange(self.N), np.arange(self.N)] = np.inf  # Ignore self-distances
        nearest_distances = np.min(distances, axis=1)  # Nearest neighbor for each ion
        d_avg = np.mean(nearest_distances)  # Average over all ions
        
        return d_avg 
    
    def get_energy(self):
        """
        Returns the total potential energy of the system.

        Includes:
            - Harmonic confinement energy
            - Coulomb repulsion between all particle pairs
        """
        
        x_vars = self.positions[:,0]  
        y_vars = self.positions[:,1]

        # Compute the first term: 1/2 * (y_i^2 + (omega_z^2 / omega_y^2) * z_i^2)
        term1 = self.half_m * (self.w**2) * np.sum(x_vars**2 + (self.g*y_vars)**2)
        
        # Compute the second term: sum of 1/|r_i - r_j| for i < j
        term2 = 0.0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                r_i = np.array([x_vars[i], y_vars[i]])  # Position of the i-th particle
                r_j = np.array([x_vars[j], y_vars[j]])  # Position of the j-th particle
                distance = np.linalg.norm(r_i - r_j)  # Euclidean distance
                if distance != 0:  # Avoid division by zero
                    term2 += 1 / distance

        return term1 + units.k * term2
    
    def get_kinetic_energy(self):

        return self.half_m * np.sum(self.velocities ** 2)
    
    def get_temperature(self):
        
        kinetic_energy = self.get_kinetic_energy()
        
        return kinetic_energy / (1 * units.kB * self.N)
    
    def get_thermal_velocities(self):
        """
        Generates and returns a (N, 2) array of random velocities scaled to match the system's temperature.

        Returns:
            np.ndarray: Thermal velocity vectors for all particles.
        """
        
        # Initialize velocities to reflect the desired temperature
        velocities = np.random.randn(self.N, 2)  # Random velocities
    
        # Scale velocities to match the temperature
        desired_kinetic_energy = 1 * units.kB * self.N * self.T # 1 = 1/2 * (# dimensions = 2) is the prefactor
        current_kinetic_energy = 0.5 * self.m * np.sum(velocities ** 2)
        scaling_factor = np.sqrt(desired_kinetic_energy / current_kinetic_energy)
        velocities *= scaling_factor
    
        return velocities
        
    #############################################################
    #############################################################
    #############################################################
    # Forcing functions
    #############################################################
    #############################################################
    #############################################################  

    def compute_forces(self):
        """
        Computes and returns the total force on each particle.

        Contributions include:
            - Coulomb repulsion between particles
            - Harmonic confinement
            - Laser cooling forces
            - Langevin thermal noise (if enabled)
            - Damping (if enabled)

        Returns:
            np.ndarray: (N, 2) array of total forces on each particle.
        """
        
        pos = self.positions
        vel = self.velocities
        N = self.N

        # Coulomb forces
        ########################################
        # compute_coulomb_forces_numba IS FASTEST FOR LARGE NUMBERS OF PARTICLES
        # compute_coulomb_forces_numpy IS FASTEST FOR SMALL NUMBERS OF PARTICLES
        
#         pairwise_forces = compute_coulomb_forces_numba(pos) # 1.69s for N = 6, 1.8s for N = 100
#         pairwise_forces = compute_coulomb_forces_numba_symmetric(pos) # 1.60 for N = 6, # 2s for N = 100
        pairwise_forces = compute_coulomb_forces_numpy(pos) # 1.0469460487365723 # about 7s for N = 100
#         pairwise_forces = compute_coulomb_forces_numpy_numba(pos) # About 6.7s for N = 100
    
        # Harmonic confinement forces
        ##########################################
        harmonic_forces = -self.m * pos * np.array([self.wx2, self.wy2])

        # Laser cooling forces
        ##############################################
        cooling_forces = np.zeros_like(self.velocities)
        for laser in self.lasers:
            cooling_forces += laser.get_force(self.velocities)
    
        # Langevin heat bath forces
        ########################################
        if self.langevin_temperature:
            # Langevin noise: sqrt(2 * gamma * k_B * T / dt) * random_gaussian

            if self.T > 0:
                sigma = np.sqrt(2 * units.kB * self.damping_parameter * self.T / self.dt)  # note dt is critical
                noise_forces = sigma * np.random.normal(size=(N, 2))
                
            else:
                noise_forces = 0.
                
        else:
            noise_forces = 0.
         
        
        if self.damping:
            damping_forces = -self.damping_parameter * self.velocities
            
        else:
            damping_forces = 0.

        
        return pairwise_forces + harmonic_forces + cooling_forces + noise_forces + damping_forces

    
    
    
    #############################################################
    #############################################################
    #############################################################
    # Initialization FUNCTIONS
    #############################################################
    #############################################################
    #############################################################
    
    def initialize_circle_positions(self):
                
        angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
    
        x0 = utility.base_radius(self.N, self.m, self.w, units.k) * np.column_stack((np.cos(angles),np.sin(angles)))
    
        self.positions = x0
        self.initial_positions = self.positions
        
        self.initialized = True
        
    def initialize_random_positions(self):
    
        radius = utility.base_radius(self.N, self.m, self.w, units.k)
    
        positions = []
        while len(positions) < self.N:
            # Generate more than needed to reduce loop iterations
            points = np.random.uniform(-radius, radius, size=(self.N * 2, 2))
            # Calculate distances from origin
            distances = np.linalg.norm(points, axis=1)
            # Filter points within the circle
            inside_circle = points[distances <= radius]
            # Add to our positions list
            positions.extend(inside_circle.tolist())
        
        # Return only the first N points as a NumPy array
        self.positions = np.array(positions[:self.N])
        self.initial_positions = self.positions
        
        self.initialized = True
        
    def check_initialization(self):
        if not self.initialized:
            self.initialize_random_positions()
            print("No initial configuration specified, initializing random configuration")
            
    #############################################################
    #############################################################
    #############################################################
    # SAVING and LOADING
    #############################################################
    #############################################################
    #############################################################   
    
    def save(self, name, filename=None):
        """
        Saves a specified simulation attribute (e.g., positions or velocities) to a JSON file.

        Args:
            name (str): The attribute name key (must be in self._attribute_map).
            filename (str, optional): Desired output filename. If None, a unique name is generated.

        Raises:
            ValueError: If the name is invalid or the attribute is None.
        """
        
        if name not in self._attribute_map:
            raise ValueError(f"Unknown save target: '{name}'")

        attr = self._attribute_map[name]['attr']
        data = getattr(self, attr)
        if data is None:
            raise ValueError(f"Attribute '{attr}' is None and cannot be saved.")

        if filename is None:
            # Generate a unique filename
            counter = 1
            while True:
                gen_name = self._attribute_map[name]['generate_filename'](self, counter)
                if not os.path.exists(gen_name):
                    filename = gen_name
                    break
                counter += 1

        with open(filename, "w") as f:
            json.dump(data.tolist(), f)

        print(f"{name.capitalize()} saved to {filename}")

    def load(self, name, filename=None):
        """
        Loads a saved simulation attribute (e.g., positions or trajectory) from a JSON file.

        Args:
            name (str): The attribute name key (must be in self._attribute_map).
            filename (str, optional): Path to the file to load. If None, a default name is used.

        Raises:
            ValueError: If the name is invalid.
        """
        
        if name not in self._attribute_map:
            raise ValueError(f"Unknown load target: '{name}'")

        attr = self._attribute_map[name]['attr']
        default_filename = self._attribute_map[name]['default_filename']
        if filename is None:
            filename = default_filename

        with open(filename, "r") as f:
            data = json.load(f)

        setattr(self, attr, np.array(data))

        if name == 'positions':
            self.initial_positions = self.positions
            self.initialized = True

        if name == 'trajectory':
            self.num_steps = self.trajectory.shape[0]
            self.N = self.trajectory.shape[1]

        print(f"{name.capitalize()} loaded from {filename}")
        
#     def save_positions(self, filename = None):
        
#         """
#         Saves positions (NumPy array) to a JSON file.

#         Parameters:
#         - positions: np.ndarray of shape (N, 2)
#         - filename: str, name of the file to save to (default triggers auto-naming)
#         """
#         if filename == None:
            
#             # Generate a unique filename 
#             counter = 1
#             while True:
#                 generated_name = f"{self.N}_{self.g}_positions_{counter}.json"
#                 if not os.path.exists(generated_name):
#                     filename = generated_name
#                     break
#                 counter += 1

#         with open(filename, "w") as f:
#             json.dump(self.positions.tolist(), f)
            
#         print(f"Positions saved to {filename}")
        
        
            
#     def load_positions(self, filename="positions.json"):
#         """
#         Loads positions from a JSON file into a NumPy array.

#         Parameters:
#         - filename: str, name of the file to load from

#         Returns:
#         - positions: np.ndarray of shape (N, 2)
#         """
#         with open(filename, "r") as f:
#             data = json.load(f)
            
#         self.positions = np.array(data)
#         self.initial_positions = self.positions
        
#         self.initialized = True
        
#     def save_trajectory(self, filename="trajectory.json"):
#         """
#         Saves the trajectory (NumPy array) to a JSON file.
#         """
#         if filename == "trajectory.json":
#             # Generate a unique filename like "trajectory of {N} particles 1.json", etc.
#             counter = 1
#             while True:
#                 generated_name = f"trajectory of {self.N} particles {counter}.json"
#                 if not os.path.exists(generated_name):
#                     filename = generated_name
#                     break
#                 counter += 1

#         with open(filename, "w") as f:
#             json.dump(self.trajectory.tolist(), f)
#         print(f"Trajectory saved to {filename}")
        
#     def save_temperature(self, filename="temperature.json"):
#         """
#         Saves the trajectory (NumPy array) to a JSON file.
#         """
#         if filename == "temperature.json":
#             # Generate a unique filename like "trajectory of {N} particles 1.json", etc.
#             counter = 1
#             while True:
#                 generated_name = f"temperature of {self.N} particles {counter}.json"
#                 if not os.path.exists(generated_name):
#                     filename = generated_name
#                     break
#                 counter += 1

#         with open(filename, "w") as f:
#             json.dump(self.temperatures.tolist(), f)
#         print(f"Trajectory saved to {filename}")

#     def load_trajectory(self, filename="trajectory.json"):
#         """
#         Loads the trajectory from a JSON file into a NumPy array.
#         """
        
        
#         with open(filename, "r") as f:
#             data = json.load(f)

#         self.trajectory = np.array(data)
#         self.num_steps = self.trajectory.shape[0]
#         self.N = self.trajectory.shape[1]
#         print(f"Trajectory loaded from {filename} of length {self.num_steps}")

         
    #############################################################
    #############################################################
    #############################################################
    # DISPLAY FUNCTIONS
    #############################################################
    #############################################################
    #############################################################
        
    def visualize(self, square = True, save = False, filename = None):
        """
        Plots the current 2D positions of particles, optionally saving the figure.

        Args:
            square (bool): If True, enforces equal scaling on both axes.
            save (bool): If True, saves the plot to a file.
            filename (str, optional): Custom filename for saving. Auto-generated if None.
        """

        
        x = self.positions[:,0]
        y = self.positions[:,1]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, color='blue', s=100, label='Particles')
    
        # Annotate each particle with its index
        for i, (y, z) in enumerate(zip(x, y)):
            plt.text(y, z, f"{i+1}", fontsize=12, ha='right', color='red')
    
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add horizontal axis
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Add vertical axis

        plt.xlabel('x-coordinate (μm)')
        plt.ylabel('y-coordinate (μm)')
        plt.title(f'Particle Positions in 2D Space for {self.N} particles')
        plt.grid(True)
        plt.legend()

        if square:
            plt.gca().set_aspect('equal', adjustable='box')
        
        if save:
            if filename == None:
                filename = f"{self.N}_{self.g}_image_with_energy_{self.get_energy()}_kappa.png"
            plt.savefig(filename)
            
        plt.show()

        
    def visualize_trajectory(self, square = True, from_index = 0, last_run = False):
        """
        Plots particle trajectories over time from the trajectory array.

        Args:
            square (bool): If True, sets equal aspect ratio.
            from_index (int): Index in trajectory to start plotting from.
            last_run (bool): If True, starts plotting from the last simulation run.
        """
        
        if last_run:
            from_index = len(self.trajectory) - self.num_steps
        
        initial_positions = self.trajectory[from_index]
        plt.figure(figsize=(8, 8))
        for i in range(self.N):
            plt.plot(self.trajectory[from_index:, i, 0], self.trajectory[from_index:, i, 1])
            plt.scatter(self.trajectory[from_index, i, 0], self.trajectory[from_index, i, 1], color='red', label=f'Start {i+1}' if i == 0 else "")
            plt.scatter(self.trajectory[-1, i, 0], self.trajectory[-1, i, 1], color='blue', label=f'End {i+1}' if i == 0 else "")

        plt.title(f"Trajectories of {self.N} Particles")
        plt.xlabel("x-coordinate (μm)")
        plt.ylabel("y-coordinate (μm)")
        plt.legend()
        plt.grid()

        if square:
            plt.gca().set_aspect('equal', adjustable='box')

        plt.show()
        
    
    def plot_density_map(self, bins=100, range=None, cmap='Greys', zoom=1, square = True, save = False):
        """
        Plots a density map of ion positions with a circle and cross at (0,0),
        regardless of where the data is concentrated.

        Parameters:
        - bins: Histogram bins (int or [int, int])
        - range: [[xmin, xmax], [ymin, ymax]] axes limits (optional)
        - cmap: Colormap (default 'Greys')
        - zoom: Ignored unless you want to implement later
        - length: How many time steps to include from trajectory
        """

        # Get trajectory positions
        positions = self.trajectory.reshape(-1, 2)
        x = positions[:, 0]
        y = positions[:, 1]

        # If range not specified, use full data bounds
        if range is None:
            buffer = 0.05  # small padding
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)
            xrange = xmax - xmin
            yrange = ymax - ymin
            range = [
                [xmin - buffer * xrange, xmax + buffer * xrange],
                [ymin - buffer * yrange, ymax + buffer * yrange]
            ]

        # Histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
        hist = hist.T  # transpose for correct orientation
        hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist

        # Plotting extent using edges
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(
            hist_norm,
            origin='lower',
            cmap=cmap,
            extent=extent,
            interpolation='nearest'
        )

        # Rough estimate for the radius of a circle/elipse to draw
        radius = utility.base_radius(self.N, self.m, self.w, units.k)

        # Plot an ellipse with radius such that radius = (a+b)/2 and such that it is a level surface of
        # the potential V = x^2 + (g y)^2
        a = 2 * radius * self.g / (self.g + 1)
        b = 2 * radius / (self.g + 1)
        ax.add_patch(Ellipse((0, 0), width=2*a, height=2*b, edgecolor='black', facecolor='none', linewidth=1))
        
        # Always draw circle and cross at (0, 0)
        ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)

        # Formatting
        ax.set_xlabel('x position (μm)')
        ax.set_ylabel('y position (μm)')
        ax.set_title('Ion Density Map')
        
        if square:
            ax.set_aspect('equal')  # square pixels
            
        fig.colorbar(im, ax=ax, label='Normalized density')
        
        if save:
            base_name = "image"
            extension = ".png"
            i = 1
            while os.path.exists(f"{base_name} {i}{extension}"):
                i += 1
            filename = f"{base_name} {i}{extension}"
            plt.savefig(filename)

        else: 
            plt.show()
        
        
    def plot_energy(self):
        """
        Plot kinetic, potential, and total energy as functions of time over the simulation.
        """
        kinetic_energies = np.zeros(self.num_steps)
        potential_energies = np.zeros(self.num_steps)
        total_energies = np.zeros(self.num_steps)

        velocities = np.copy(self.velocities)
        positions = np.copy(self.trajectory[0])

        for step in range(self.num_steps):
            positions = self.trajectory[step]

            # Estimate velocities using finite difference (central difference)
            if 1 <= step < self.num_steps - 1:
                v_est = (self.trajectory[step + 1] - self.trajectory[step - 1]) / (2 * self.dt)
            elif step == 0:
                v_est = (self.trajectory[step + 1] - self.trajectory[step]) / self.dt
            else:
                v_est = (self.trajectory[step] - self.trajectory[step - 1]) / self.dt

            # Kinetic energy
            ke = 0.5 * self.m * np.sum(v_est ** 2)
            kinetic_energies[step] = ke

            # Potential energy
            self.positions = positions
            pe = self.get_energy()
            potential_energies[step] = pe

            # Total energy
            total_energies[step] = ke + pe

        times = np.arange(self.num_steps) * self.dt

        plt.figure(figsize=(10, 6))
        plt.plot(times, kinetic_energies, label="Kinetic Energy")
        plt.plot(times, potential_energies, label="Potential Energy")
        plt.plot(times, total_energies, label="Total Energy", linestyle='--')
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Energies Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    #############################################################
    #############################################################
    #############################################################
    # RUNNING FUNCTIONS
    #############################################################
    #############################################################
    #############################################################
    
    def timestep(self, step):
        dt = self.dt
        m = self.m
        
        #Compute forces
        forces = self.compute_forces()

        # Update positions using Verlet integration
        new_positions = self.positions + dt * self.velocities + 0.5 * (forces / m) * dt**2

        # Update velocities
        self.velocities += (forces / m) * dt

        # Store positions
        self.trajectory[step] = self.positions
        self.velocity_trajectory[step] = self.velocities

        # Update positions for the next step
        self.positions = new_positions

    
    def run(self, compute_new_velocities = True, verbose = False, again = False):
        start = time()
        self.check_initialization()
        shift = 0
        
        print(f"Simulation running for {self.simulation_time} μs")
        
        if again: 
            self.simulation_time += self.num_steps * self.dt
        
        if again and (np.any(self.trajectory)): # I.e. as long as self.trajectory isn't all zeros
            
            num_steps_done = len(self.trajectory)
            shift = num_steps_done
            compute_new_velocities = False
            
            self.trajectory = np.concatenate((self.trajectory, np.zeros((self.num_steps, *self.trajectory.shape[1:]))), dtype=self.trajectory.dtype, axis=0)
            self.velocity_trajectory = np.concatenate((self.trajectory, np.zeros((self.num_steps, *self.velocity_trajectory.shape[1:]))), dtype=self.velocity_trajectory.dtype, axis=0)
                
            if verbose:
                                
                self.temperatures = np.concatenate((self.temperatures, np.zeros(self.num_steps // self.grainyness)), axis = 0)
                self.potential_energies = np.concatenate((self.potential_energies, np.zeros(self.num_steps // self.grainyness)), axis = 0)
                self.kinetic_energies = np.concatenate((self.kinetic_energies, np.zeros(self.num_steps // self.grainyness)), axis = 0)
                self.total_energies = np.concatenate((self.total_energies, np.zeros(self.num_steps // self.grainyness)), axis = 0)
                
                self.times = np.concatenate((self.times, np.zeros(self.num_steps // self.grainyness)), axis = 0)
            
            
        
        if compute_new_velocities:       
            self.velocities = self.get_thermal_velocities()

        if not verbose:
            
            # Main simulation loop
            for step in range(shift, shift + self.num_steps):

                self.timestep(step)
                
        else:
            
            i = shift // self.grainyness
            for step in range(shift, shift + self.num_steps):

                self.timestep(step)
                
                if step % self.grainyness == 0:
                    
                    self.temperatures[i] = self.get_temperature()
                    self.potential_energies[i] = self.get_energy()
                    self.kinetic_energies[i] = self.get_kinetic_energy()
                    self.total_energies[i] = self.get_energy() + self.get_kinetic_energy()
                    self.times[i] = step * self.dt
                    i += 1

            
            plt.plot(self.times, units.theta_to_mK(self.temperatures), label = "Temperature")
            plt.xlabel("Time (μs)")
            plt.ylabel("Temperature (mK)")
            plt.show()

            plt.plot(self.times, 1e9 * units.kappa_to_neV(self.potential_energies), label = "Potential energy")
            plt.plot(self.times, 1e9 * units.kappa_to_neV(self.kinetic_energies), label = "Kinetic energy")
            plt.plot(self.times, 1e9 * units.kappa_to_neV(self.total_energies), label = "Total energy")
            plt.xlabel("Time (μs)")
            plt.ylabel("Energy (neV)")
            plt.legend()
            plt.show()
            
    def walk_to_lower_energy(self):
        
        self.check_initialization()
        
        velocities = self.get_thermal_velocities()

        # Forces placeholder
        forces = np.zeros((self.N, 2))

        # Storage for positions
        self.trajectory = np.zeros((self.num_steps, self.N, 2))

        # Main simulation loop
        for step in range(self.num_steps):
            initial_energy = self.get_energy()
            
            #Compute forces
            forces = self.compute_forces()

            # Update positions using Verlet integration
            new_positions = self.positions + self.dt * velocities + 0.5 * (forces / self.m) * self.dt**2

            # Update velocities
            velocities += (forces / self.m) * self.dt
            
            old_positions = self.positions
            
            # Update positions for the next step
            self.positions = new_positions
            
            if self.get_energy() > initial_energy:
                velocities = self.get_thermal_velocities()
                self.trajectory[step] = old_positions
                
            else:
            
                # Store positions
                self.trajectory[step] = new_positions
            
        
    def find_equilibrium(self):
        vars = np.random.rand(2 * self.N) * 10
        y_vars = vars[:self.N]  # First N variables are y_i
        z_vars = vars[self.N:]   # Next N variables are z_i
        result = minimize(utility.objective_function, vars, args=(self.N, self.g, self.m, self.w, units.k), method='BFGS')

        self.positions = utility.vars_to_positions(result.x)
        
        
    def full_freezeout(self):
        
        self.check_initialization()

        # Forces placeholder
        forces = np.zeros((self.N, 2))

        # Storage for positions
        self.trajectory = np.zeros((self.num_steps, self.N, 2))

        velocities = np.zeros((self.N, 2))

        # Main simulation loop
        for step in range(self.num_steps):
            # Compute forces
            forces = self.compute_forces()

            # Update positions using Verlet integration
            new_positions = self.positions + self.dt * velocities + 0.5 * (forces / self.m) * self.dt**2

            # Update velocities
            velocities = (forces / self.m) * self.dt

            # Store positions
            self.trajectory[step] = self.positions

            # Update positions for the next step
            self.positions = new_positions
            
            
    def partial_freezeout(self):
        
        self.check_initialization()
        
        initial_temperature = self.T

        temperature_step = initial_temperature/self.num_steps

        # Forces placeholder
        forces = np.zeros((self.N, 2))

        # Storage for positions
        self.trajectory = np.zeros((self.num_steps, self.N, 2))

        velocities = np.zeros((self.N,2))

        # Main simulation loop
        for step in range(self.num_steps):
            # Compute forces
            forces = self.compute_forces()

            # Update positions using Verlet integration
            new_positions = self.positions + self.dt * velocities + 0.5 * (forces / self.m) * self.dt**2

            # Update velocities
            velocities = (forces / self.m) * self.dt

            velocities += self.get_thermal_velocities()

            self.T -= temperature_step

            # Store positions
            self.trajectory[step] = self.positions

            # Update positions for the next step
            self.positions = new_positions

        self.T = initial_temperature
            
    def minimize_force(self):
        
        self.check_initialization()    
        
        sol = root(utility.fixed_point_equation, self.positions, args = (self.g, self.w, units.k, self.m))
    
        if sol.success:
            self.positions = sol.x.reshape((self.N,2))

        else:
            raise RuntimeError("Root finding did not converge: " + sol.message)
    
    def check_stability(self):
        initial_temp = self.T
        self.T = 0
        initial_positions = self.positions
        initial_trajectory = self.trajectory
        
        
        self.run()
        final_positions = self.positions
        
        self.T = initial_temp
        self.positions = initial_positions
        self.trajectory = initial_trajectory
        
        return np.linalg.norm(initial_positions-final_positions)
    
    def compare_to_metastable(self, metastable_state, compute_new_velocities = False):
        
        deviations = np.zeros(self.num_steps)
        
        with open(metastable_state, "r") as f:
            data = json.load(f)
            
        metastable_positions = np.array(data)
                
        self.check_initialization()
        
        # i.e. if we A) need to compute new velocities or B) the velocities are all zero, then compute new velocities
        if compute_new_velocities or (not np.any(self.velocities)): 
                        
            self.velocities = self.get_thermal_velocities()
        
        # Forces placeholder
        forces = np.zeros((self.N, 2))

        # Storage for positions
        self.trajectory = np.zeros((self.num_steps, self.N, 2))

        # Main simulation loop
        for step in range(self.num_steps):
            
            #Compute forces
            forces = self.compute_forces()

            # Update positions using Verlet integration
            new_positions = self.positions + self.dt * self.velocities + 0.5 * (forces / self.m) * self.dt**2

            # Update velocities
            self.velocities += (forces / self.m) * self.dt

            # Store positions
            self.trajectory[step] = self.positions
            deviations[step] = utility.minimum_matching_distance(metastable_positions, self.positions)
            
            # Update positions for the next step
            self.positions = new_positions
            
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(self.num_steps) * self.dt, deviations, label='Deviation from metastable state')
        plt.xlabel('Time')
        plt.ylabel('Deviation')
        plt.title('Deviation vs Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        self.deviations = deviations
    
    
###########################################################################
###########################################################################
###########################################################################
# Quenching
###########################################################################
###########################################################################
###########################################################################

    def quench(self, new_g):
        x_vars = self.positions[:,0]  
        y_vars = self.positions[:,1]        
        
        self.check_initialization()
        initial_potential_energy = self.half_m * (self.w**2) * np.sum(x_vars**2 + (self.g*y_vars)**2)
        
        print(f"Quenching. Initial gamma: {self.g}, final gamma: {new_g}")
        self.g = new_g

        would_be_new_potential_energy = 0.5 * self.m * (self.w**2) * np.sum(x_vars**2 + (self.g*y_vars)**2)
        
        new_w = self.w * np.sqrt(initial_potential_energy/would_be_new_potential_energy)
        print(f"Initial w: {self.w}, final w: {new_w}")
        self.w = new_w
        
        
#         if conserve_energy:
#             new_potential_energy = self.half_m * (self.w**2) * np.sum(x_vars**2 + (self.g*y_vars)**2)
#             desired_kinetic_energy = initial_potential_energy + initial_kinetic_energy - new_potential_energy
#             assert desired_kinetic_energy > 0, "Desired kinetic energy is negative!"
#             new_T = desired_kinetic_energy/(1 * self.N)
#             print(f"Temperature updated from {self.T} to {new_T}")
#             self.T = new_T

###########################################################################
###########################################################################
###########################################################################
# Optimization
###########################################################################
###########################################################################
###########################################################################

@numba.njit(parallel=True)
def compute_coulomb_forces_numba(pos):
    N = pos.shape[0]
    forces = np.zeros((N, 2))
    for i in numba.prange(N):
        for j in range(N):
            if i != j:
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                r2 = dx * dx + dy * dy
                r3_inv = 1.0 / (r2 * np.sqrt(r2))
                forces[i, 0] += dx * r3_inv
                forces[i, 1] += dy * r3_inv
    return units.k * forces

@numba.njit(parallel=True)
def compute_coulomb_forces_numba_symmetric(pos):
    N = pos.shape[0]
    forces = np.zeros((N, 2))
    for i in numba.prange(N):
        for j in range(i + 1, N):  # Only compute i < j
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r2 = dx * dx + dy * dy + 1e-12  # small epsilon to prevent div by zero
            r3_inv = 1.0 / (r2 * np.sqrt(r2))
            fx = dx * r3_inv
            fy = dy * r3_inv
            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[j, 0] -= fx  # Newton's third law
            forces[j, 1] -= fy
    return units.k * forces

# @numba.njit(parallel=True)
def compute_coulomb_forces_numpy(pos):
    # pos: shape (N, 2)
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # shape (N, N, 2)
    r2 = np.sum(delta**2, axis=-1) + 1e-12  # shape (N, N), add eps to avoid zero
    np.fill_diagonal(r2, np.inf)  # set self-interaction to zero force
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))  # shape (N, N)
    forces = np.sum(delta * inv_r3[:, :, np.newaxis], axis=1)  # shape (N, 2)
    return units.k * forces

@numba.njit(parallel=True)
def compute_coulomb_forces_numpy_numba(pos):
    N = pos.shape[0]
    dim = pos.shape[1]
    delta = np.zeros((N, N, dim))
    for i in range(N):
        for j in range(N):
            for d in range(dim):
                delta[i, j, d] = pos[i, d] - pos[j, d]

    r2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for d in range(dim):
                r2[i, j] += delta[i, j, d] ** 2
            if i == j:
                r2[i, j] = np.inf
            else:
                r2[i, j] += 1e-12  # avoid divide-by-zero

    inv_r3 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            inv_r3[i, j] = 1.0 / (r2[i, j] * r2[i, j]**0.5)

    forces = np.zeros((N, dim))
    for i in range(N):
        for j in range(N):
            for d in range(dim):
                forces[i, d] += delta[i, j, d] * inv_r3[i, j]

    return units.k * forces


@numba.njit
def doppler_force_numba(velocities, k_vec, saturation, detuning, Gamma, hbar):
    # velocities: (N, 2), k_vec: (2,)
    N = velocities.shape[0]

    # Doppler shift: (N,)
    doppler_shift = np.dot(velocities, k_vec)

    # Effective detuning: (N,)
    effective_detuning = detuning - doppler_shift

    # Scattering rate (N,)
    denom = 1 + saturation + (2 * effective_detuning / Gamma) ** 2
    scattering_rate = 0.5 * Gamma * saturation / denom

    # Force: (N, 2)
    forces = np.zeros((N, 2))
    for i in range(N):
        for j in range(2):
            forces[i, j] = hbar * scattering_rate[i] * k_vec[j]

    return forces

###########################################################################
###########################################################################
###########################################################################
# General functions
###########################################################################
###########################################################################
###########################################################################

########################################################################################
########################################################################################
########################################################################################
# Quenching functions
########################################################################################
########################################################################################
########################################################################################

def generate_quench_animation(N, w, g, m, num_steps, loadfile, trajectory_folder, g_start, g_end, g_step, T = 0, dt = 1e-2, damping = False, damping_parameter = 1, langevin_temperature = False, lasers = None):
    
    os.makedirs(trajectory_folder, exist_ok = True)
    
    S = Simulation(
        N = N,
        w = w,
        g = g,
        m = m,
        T = T,
        dt = dt,
        num_steps = num_steps,
        damping = damping,
        damping_parameter = damping_parameter,
        langevin_temperature = langevin_temperature,
        lasers = lasers
        )
    
    S.load("positions",loadfile)
    
    gammas = np.linspace(g_start, g_end, g_step)
    
    
    
    for gamma in gammas:
        
        traj_file = f"{trajectory_folder}/{N}_{gamma}_traj_{num_steps}_steps.json"
        
        # Skip if file already exists
        if os.path.exists(traj_file):
            print(f"Skipping gamma = {gamma}, file already exists.")
            continue
        
        S.reset()
        S.g = 1
        S.w = 1
        S.quench(gamma)
        print(f"Running gamma = {gamma}")
        S.run()
#       S.plot_density_map(save = True)
#         S.save_trajectory(traj_file)
        S.save("trajectory",traj_file)
    
    # Write simulation parameters to -details.txt
    details_path = os.path.join(trajectory_folder, "-details.txt")
    if not os.path.exists(details_path):
        with open(details_path, "w") as f:
            f.write("Simulation Parameters:\n")
            f.write(f"N = {N}\n")
            f.write(f"w = {w}\n")
            f.write(f"g = {g}\n")
            f.write(f"m = {m}\n")
            f.write(f"T = {T}\n")
            f.write(f"dt = {dt}\n")
            f.write(f"num_steps = {num_steps}\n")
            f.write(f"damping = {damping}\n")
            f.write(f"damping_parameter = {damping_parameter}\n")
            f.write(f"langevin_temperature = {langevin_temperature}\n")
            f.write(f"g_start = {g_start}\n")
            f.write(f"g_end = {g_end}\n")
            f.write(f"g_step = {g_step}\n")
    
    print("Done!")
        
def process_json_arrays(trajectory_folder, animation_folder, range = None, numsteps = None):
    """
    Processes all JSON files in a folder, assuming each file contains an (num_steps, N, 2) array.
    
    Args:
        trajectory_folder (str): Path to the folder containing JSON files.
    """
    os.makedirs(animation_folder, exist_ok = True)
    
    for filename in os.listdir(trajectory_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(trajectory_folder, filename)
            
            
            image_path = f"{animation_folder}/{filename} 1.png"

            if os.path.exists(image_path):
                print(f"Skipping gamma = {get_gamma_from_filename(filename)}, image already compiled.")
                continue
            
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    
                    # Assuming the JSON contains a nested list representing a (x, N, 2) array
                    arr = np.array(data)
                    
                    if range == None:
                        range = compute_range_from_array(arr[0])
                    
                    if (numsteps == None) or (numsteps == get_numsteps_from_filename(filename)):
                        save_density_map(arr, name = filename, save_folder = animation_folder, range = range, gamma = get_gamma_from_filename(filename))
                                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
                    
    print("Done processing trajectories to images!")
                    
def compute_range_from_array(array):
    positions = array.reshape(-1, 2)
    x = positions[:, 0]
    y = positions[:, 1]
    
    buffer = 0.05  # small padding
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xrange = xmax - xmin
    yrange = ymax - ymin

    return [
        [xmin - buffer * xrange, xmax + buffer * xrange],
        [ymin - buffer * yrange, ymax + buffer * yrange]
    ]


                    
def save_density_map(trajectory, bins=100, save_folder = None, range=None, cmap='Greys', zoom=1, save = True, name = "image", gamma = None):

    # Get trajectory positions
    positions = trajectory.reshape(-1, 2)
    x = positions[:, 0]
    y = positions[:, 1]

    # If range not specified, use full data bounds
    if range is None:
        buffer = 0.05  # small padding
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        xrange = xmax - xmin
        yrange = ymax - ymin
        range = [
            [xmin - buffer * xrange, xmax + buffer * xrange],
            [ymin - buffer * yrange, ymax + buffer * yrange]
        ]

    # Histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
    hist = hist.T  # transpose for correct orientation
    hist_norm = hist / np.max(hist) if np.max(hist) > 0 else hist

    # Plotting extent using edges
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        hist_norm,
        origin='lower',
        cmap=cmap,
        extent=extent,
        interpolation='nearest'
    )

# #         Rough estimate for the radius of a circle/elipse to draw
#         radius = base_radius(self.N, self.m, self.w, self.k)

#         # Plot an ellipse with radius such that radius = (a+b)/2 and such that it is a level surface of
#         # the potential V = x^2 + (g y)^2
#         a = 2 * radius * self.g / (self.g + 1)
#         b = 2 * radius / (self.g + 1)
#         ax.add_patch(Ellipse((0, 0), width=2*a, height=2*b, edgecolor='black', facecolor='none', linewidth=1))

    # Always draw circle and cross at (0, 0)
    ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)

    # Formatting
    ax.set_xlabel('$x$ position')
    ax.set_ylabel('$y$ position')
    
    ax.set_title('Ion Density Map. Quenched to \n$\gamma$' + f' = {gamma}')

    ax.set_aspect('equal')  # square pixels

    fig.colorbar(im, ax=ax, label='Normalized density')

    if save:
        folder = save_folder
        base_name = name
        extension = ".png"
        i = 1
        while os.path.exists(f"{folder}/{base_name} {i}{extension}"):
            i += 1
        filename = f"{folder}/{base_name} {i}{extension}"
        plt.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")

#     else: 
#         plt.show()

def get_gamma_from_filename(filename):
    """
    Extracts the gamma value from a filename like '6_0.8666666666666667_traj_10000_steps.json'.
    
    Args:
        filename (str): The filename to parse.
        
    Returns:
        float: The extracted gamma value.
        
    Raises:
        ValueError: If gamma could not be parsed.
    """
    try:
        parts = filename.split('_')
        gamma_str = parts[1]  # Index 1 is the gamma part
        return float(gamma_str)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not extract gamma from filename '{filename}': {e}")
        
def get_numsteps_from_filename(filename):
    """
    Extracts the gamma value from a filename like '6_0.8666666666666667_traj_10000_steps.json'.
    
    Args:
        filename (str): The filename to parse.
        
    Returns:
        float: The extracted gamma value.
        
    Raises:
        ValueError: If gamma could not be parsed.
    """
    try:
        parts = filename.split('_')
        gamma_str = parts[3]  # Index 3 is the numsteps part
        return float(gamma_str)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not extract gamma from filename '{filename}': {e}")

    

def create_animation_from_images(animation_folder, output_path, fps=5, reverse=False):
    """
    Creates an animation from PNG images sorted by gamma value extracted from filenames.

    Args:
        animation_folder (str): Path to folder with .png files.
        output_path (str): Output .gif or .mp4 path.
        fps (int): Frames per second.
        reverse (bool): If True, reverse the order of playback.
    """
    gamma_file_pairs = []

    for filename in os.listdir(animation_folder):
        if filename.endswith(".png"):
            try:
                parts = filename.split('_')
                gamma = float(parts[1])  # extract gamma from second part
                gamma_file_pairs.append((gamma, filename))
            except (IndexError, ValueError):
                print(f"Skipping file due to parse error: {filename}")

    gamma_file_pairs.sort(reverse=reverse)
    images = []
    for gamma, fname in gamma_file_pairs:
        img_path = os.path.join(animation_folder, fname)
        images.append(imageio.imread(img_path))

    ext = os.path.splitext(output_path)[-1].lower()

    if ext == '.gif':
        imageio.mimsave(output_path, images, fps=fps)
    elif ext == '.mp4':
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            for img in images:
                writer.append_data(img)
    else:
        raise ValueError("Unsupported output format. Use .gif or .mp4")

    print(f"Saved animation to {output_path}")
