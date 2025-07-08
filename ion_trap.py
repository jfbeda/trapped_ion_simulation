
import numpy as np
import os
from scipy.optimize import minimize
from scipy.optimize import root
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.patches import Circle # No longer needed?
from matplotlib.patches import Ellipse
import time
import json
import matplotlib.animation as animation
import math

import imageio.v2 as imageio
from natsort import natsorted  # Optional, to sort numerically

class Simulation:
    
    def __init__(self, N, w, g, k, m, T, dt, num_steps):
        self.N = N
        self.w = w
        self.g = g
        self.k = k
        self.m = m
        self.T = T
        self.dt = dt
        self.num_steps = num_steps
        self.wx = w
        self.wy = w * g
        self.initialized = False
        self.initial_positions = np.zeros((self.N, 2))
        self.positions = self.initial_positions
        self.velocities = np.zeros((self.N, 2))
        self.trajectory = np.zeros((self.num_steps, self.N, 2))
    
    def __str__(self):
        potential_energy = self.get_energy()
        kinetic_energy = 0.5 * self.m * np.sum(self.velocities **2)
        
        string = "Simulation object"
        
        if self.initialized:
            string += f"\nN = {self.N}, g = {self.g}, T = {self.T}"
            string += f"\nPotential energy = {potential_energy}"
            string += f"\nPotential energy per particle = {potential_energy/self.N}"
            string += f"\nKinetic energy = {kinetic_energy}"
            string += f"\nTotal energy = {kinetic_energy + potential_energy}"
            string += f"\nRMS force: {np.linalg.norm(fixed_point_equation(positions_to_flattened_positions(self.positions), self.g, self.w, self.k, self.m))}"

            
            temp_T = self.T
            temp_num_steps = self.num_steps
            self.T = 1e-3 #Can be adjusted
            self.num_steps = 100
            
            string += f"\nStability: {self.check_stability()} (RMS displacement after running {self.num_steps} steps of simulation)"
            self.T = temp_T
            self.num_steps = temp_num_steps
            
            string += f"\nAverage interparticle distance = {self.get_average_interparticle_distance()}"
#            string += f"\nAverage distance from origin = {self.get_average_distance_from_center()}"
        
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
        # Returns average distance from origin of all the particles
        return np.mean(np.linalg.norm(self.positions, axis=1))
    
    def get_average_interparticle_distance(self):
        """
        Compute the average nearest-neighbor distance in the initial configuration.

        Parameters:
        - initial_positions: (N, 2) array of initial (y, z) positions.

        Returns:
        - d_avg: Average nearest-neighbor distance.
        """
        distances = np.linalg.norm(self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :], axis=2)
        distances[np.arange(self.N), np.arange(self.N)] = np.inf  # Ignore self-distances
        nearest_distances = np.min(distances, axis=1)  # Nearest neighbor for each ion
        d_avg = np.mean(nearest_distances)  # Average over all ions
        
        return d_avg 
    
    def get_energy(self):
        x_vars = self.positions[:,0]  
        y_vars = self.positions[:,1]

        # Compute the first term: 1/2 * (y_i^2 + (omega_z^2 / omega_y^2) * z_i^2)
        term1 = 0.5 * self.m * (self.w**2) * np.sum(x_vars**2 + (self.g*y_vars)**2)
        
        # Compute the second term: sum of 1/|r_i - r_j| for i < j
        term2 = 0.0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                r_i = np.array([x_vars[i], y_vars[i]])  # Position of the i-th particle
                r_j = np.array([x_vars[j], y_vars[j]])  # Position of the j-th particle
                distance = np.linalg.norm(r_i - r_j)  # Euclidean distance
                if distance != 0:  # Avoid division by zero
                    term2 += 1 / distance

        return term1 + self.k*term2
    
    def get_thermal_velocities(self):
        
        # Initialize velocities to reflect the desired temperature
        velocities = np.random.randn(self.N, 2)  # Random velocities
    
        # Scale velocities to match the temperature
        current_kinetic_energy = 0.5 * self.m * np.sum(velocities**2)
        desired_kinetic_energy = 1 * self.N * self.T # 1 = 1/2 * (# dimensions = 2) is the prefactor
        scaling_factor = np.sqrt(desired_kinetic_energy / current_kinetic_energy)
        velocities *= scaling_factor
    
        return velocities
        

#     def compute_forces(self):
        
#         forces = np.zeros((self.N, 2))
#         epsilon = 1e-8  # Small value to avoid division by zero

#         # Compute all pairwise distances
#         deltas = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
#         distances = np.linalg.norm(deltas, axis=2) + epsilon

#         # Compute force magnitude for all pairs
#         inverse_distance_cubed = 1.0 / distances**3
#         np.fill_diagonal(inverse_distance_cubed, 0)  # Ignore self-interaction

#         # Compute net forces from pairwise interactions
#         for i in range(self.N):
#             forces[i] += np.sum(deltas[i] * inverse_distance_cubed[i, :, np.newaxis], axis=0)

#         # Add confinement potential forces
#         forces[:, 0] -= self.positions[:, 0]
#         forces[:, 1] -= (self.wx**2 / self.wy**2) * self.positions[:, 1]

#         return forces
    
    def compute_forces(self):
        
        forces = np.zeros((self.N,2))
        forces_flattened = -fixed_point_equation(self.positions.flatten(),self.g, self.w, self.k, self.m)
        forces[:,0] = forces_flattened[::2]
        forces[:,1] = forces_flattened[1::2]
        
        return forces
    
#     def compute_forces(self):
#         pos = self.positions
#         diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 2)
#         dist_sq = np.sum(diff**2, axis=-1) + np.eye(self.N)  # (N, N), add eye to avoid division by zero
#         inv_dist_cubed = np.where(dist_sq > 0, 1.0 / dist_sq**(1.5), 0.0)  # Avoid inf
#         force_matrix = diff * inv_dist_cubed[:, :, np.newaxis]  # (N, N, 2)
#         pairwise_forces = np.sum(force_matrix, axis=1)  # Net force on each particle

#         # Confinement potential
#         wx2, wy2 = self.w**2, (self.w * self.g)**2
#         harmonic_forces = -self.m * np.stack([wx2 * pos[:, 0], wy2 * pos[:, 1]], axis=1)

#         return self.k * pairwise_forces + harmonic_forces

    
    def fixed_point_equation(position_vectors_flat, g, w = 1., k = 1., m = 1.):
        assert position_vectors_flat.dtype == float, "The position vectors must have data type float"

        N = int(len(position_vectors_flat)/2)

        position_vectors = position_vectors_flat.reshape((N, 2))
        # Note that position_vectors_flat is of the form [x1,y1,x2,y2,x3,y3] NOT [x1,x2,x3,y1,y2,y3]
        # position_vectors is of the form [[x1,y1],[x2,y2],[x3,y3]]

        equations = np.zeros_like(position_vectors)

        # Diagonal anisotropy matrix
        Gamma = np.diag([1.0, g**2])

        for i in range(N):
            r_i = position_vectors[i]
            sum_term = np.zeros(2)

            for q in range(N):
                if q != i:
                    r_q = position_vectors[q]
                    diff = r_q - r_i
                    norm_sq = np.dot(diff, diff)
                    sum_term += (diff) / (norm_sq**(3/2))

            equations[i] = m * w**2 * (Gamma @ r_i) +k * sum_term

        return equations.flatten()
    
    #############################################################
    #############################################################
    #############################################################
    # Initialization FUNCTIONS
    #############################################################
    #############################################################
    #############################################################
    
    def initialize_circle_positions(self, initial_radius = 1.):
                
        angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)
    
        x0 = initial_radius * np.column_stack((np.cos(angles),np.sin(angles)))
    
        self.positions = x0
        self.initial_positions = self.positions
        
        self.initialized = True
        
    def initialize_random_positions(self, radius = 1.):
    
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
        
    def save_positions(self, filename = "positions.json"):
        
        """
        Saves positions (NumPy array) to a JSON file.

        Parameters:
        - positions: np.ndarray of shape (N, 2)
        - filename: str, name of the file to save to (default triggers auto-naming)
        """
        if filename == "positions.json":
            # Generate a unique filename like "positions of {N} particles 1.json", "positions of {N} particles 2.json", etc.
            counter = 1
            while True:
                generated_name = f"positions of {self.N} particles {counter}.json"
                if not os.path.exists(generated_name):
                    filename = generated_name
                    break
                counter += 1

        with open(filename, "w") as f:
            json.dump(self.positions.tolist(), f)
            
        print(f"Positions saved to {filename}")
            
    def load_positions(self, filename="positions.json"):
        """
        Loads positions from a JSON file into a NumPy array.

        Parameters:
        - filename: str, name of the file to load from

        Returns:
        - positions: np.ndarray of shape (N, 2)
        """
        with open(filename, "r") as f:
            data = json.load(f)
            
        self.positions = np.array(data)
        self.initial_positions = self.positions
        
        self.initialized = True
        
    def save_trajectory(self, filename="trajectory.json"):
        """
        Saves the trajectory (NumPy array) to a JSON file.
        """
        if filename == "trajectory.json":
            # Generate a unique filename like "trajectory of {N} particles 1.json", etc.
            counter = 1
            while True:
                generated_name = f"trajectory of {self.N} particles {counter}.json"
                if not os.path.exists(generated_name):
                    filename = generated_name
                    break
                counter += 1

        with open(filename, "w") as f:
            json.dump(self.trajectory.tolist(), f)
        print(f"Trajectory saved to {filename}")

    def load_trajectory(self, filename="trajectory.json"):
        """
        Loads the trajectory from a JSON file into a NumPy array.
        """
        
        
        with open(filename, "r") as f:
            data = json.load(f)

        self.trajectory = np.array(data)
        self.num_steps = self.trajectory.shape[0]
        self.N = self.trajectory.shape[1]
        print(f"Trajectory loaded from {filename} of length {self.num_steps}")
        
    def crop_trajectory(self, length):
        assert length <= self.num_steps, "New length parameter must be shorter than old one"
        
        self.num_steps = length
        self.trajectory = self.trajectory[:self.num_steps]

         
    #############################################################
    #############################################################
    #############################################################
    # DISPLAY FUNCTIONS
    #############################################################
    #############################################################
    #############################################################
        
    def visualize(self, square = True):
        x = self.positions[:,0]
        y = self.positions[:,1]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, color='blue', s=100, label='Particles')
    
        # Annotate each particle with its index
        for i, (y, z) in enumerate(zip(x, y)):
            plt.text(y, z, f"{i+1}", fontsize=12, ha='right', color='red')
    
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add horizontal axis
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)  # Add vertical axis

        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.title(f'Particle Positions in 2D Space for {self.N} particles')
        plt.grid(True)
        plt.legend()

        if square:
            plt.gca().set_aspect('equal', adjustable='box')

        plt.show()
        
    def visualize_trajectory(self, square = True):

        """
        Plot the trajectories of the particles.
        """
        initial_positions = self.trajectory[0]
        plt.figure(figsize=(8, 8))
        for i in range(self.N):
            plt.plot(self.trajectory[:, i, 0], self.trajectory[:, i, 1])
            plt.scatter(self.trajectory[0, i, 0], self.trajectory[0, i, 1], color='red', label=f'Start {i+1}' if i == 0 else "")
            plt.scatter(self.trajectory[-1, i, 0], self.trajectory[-1, i, 1], color='blue', label=f'End {i+1}' if i == 0 else "")

        plt.title(f"Trajectories of {self.N} Particles")
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.legend()
        plt.grid()

        if square:
            plt.gca().set_aspect('equal', adjustable='box')

        plt.show()
        
        
#     def animate_trajectory(self, save_as=None):
#         """
#         Creates an animation of ion dynamics using scatter and set_offsets.
#         """
#         import matplotlib.pyplot as plt
#         import matplotlib.animation as animation

#         fig, ax = plt.subplots(figsize=(6, 6))

#         ax.set_xlim(np.min(self.trajectory[:, :, 0]) - 0.5, np.max(self.trajectory[:, :, 0]) + 0.5)
#         ax.set_ylim(np.min(self.trajectory[:, :, 1]) - 0.5, np.max(self.trajectory[:, :, 1]) + 0.5)
#         ax.set_xlabel("x-coordinate")
#         ax.set_ylabel("y-coordinate")
#         ax.set_title("Ion Motion Over Time")

#         # Use scatter for better compatibility with animations
#         scatter = ax.scatter([], [], s=50, color='blue')

#         def update(frame):
#             scatter.set_offsets(self.trajectory[frame])
#             return scatter,

#         self.anim = animation.FuncAnimation(
#             fig, update, frames=self.num_steps, interval=50, blit=True
#         )

#         if save_as:
#             self.anim.save(save_as, writer="ffmpeg", fps=30)
#             print(f"Animation saved as {save_as}")

#         plt.show()

    def animate_trajectory(self, save_as=None):
        """
        Creates an animation of ion dynamics using scatter and set_offsets.
        Displays progress bar when saving animation.
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from tqdm import tqdm
        import matplotlib

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.set_xlim(np.min(self.trajectory[:, :, 0]) - 0.5, np.max(self.trajectory[:, :, 0]) + 0.5)
        ax.set_ylim(np.min(self.trajectory[:, :, 1]) - 0.5, np.max(self.trajectory[:, :, 1]) + 0.5)
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_title("Ion Motion Over Time")

        scatter = ax.scatter([], [], s=50, color='blue')

        def update(frame):
            scatter.set_offsets(self.trajectory[frame])
            return scatter,

        self.anim = animation.FuncAnimation(
            fig, update, frames=self.num_steps, interval=50, blit=True
        )

        if save_as:
            # Wrap the ffmpeg writer with a progress bar
            class TqdmWriter(animation.FFMpegWriter):
                def setup(self, fig, outfile, dpi, *args, **kwargs):
                    self._tqdm = tqdm(total=self.num_frames, desc="Saving animation", unit="frame")
                    super().setup(fig, outfile, dpi, *args, **kwargs)

                def grab_frame(self, **kwargs):
                    super().grab_frame(**kwargs)
                    self._tqdm.update(1)

                def finish(self):
                    super().finish()
                    self._tqdm.close()

            writer = TqdmWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
            self.anim.save(save_as, writer=writer)
            print(f"Animation saved as {save_as}")

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
        radius = base_radius(self.N)

        # Plot an ellipse with radius such that radius = (a+b)/2 and such that it is a level surface of
        # the potential V = x^2 + (g y)^2
        a = 2 * radius * self.g / (self.g + 1)
        b = 2 * radius / (self.g + 1)
        ax.add_patch(Ellipse((0, 0), width=2*a, height=2*b, edgecolor='black', facecolor='none', linewidth=1))
        
        # Always draw circle and cross at (0, 0)
        ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)

        # Formatting
        ax.set_xlabel('x podsition')
        ax.set_ylabel('y position')
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
    
    
    def run(self, compute_new_velocities = True):
        
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

            # Update positions for the next step
            self.positions = new_positions
            
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
        result = minimize(objective_function, vars, args=(self.N, self.g, self.m, self.w, self.k), method='BFGS')

        self.positions = vars_to_positions(result.x)
        
        
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
        
        sol = root(fixed_point_equation, self.positions, args = (self.g, self.w, self.k, self.m))
    
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
            deviations[step] = minimum_matching_distance(metastable_positions, self.positions)
            
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

    def quench(self, new_g, conserve_energy = True):
        x_vars = self.positions[:,0]  
        y_vars = self.positions[:,1]        
        
        self.check_initialization()
        initial_potential_energy = 0.5 * self.m * (self.w**2) * np.sum(x_vars**2 + (self.g*y_vars)**2)
        
        print(f"Quenching. Initial gamma: {self.g}, final gamma: {new_g}")
        self.g = new_g

        would_be_new_potential_energy = 0.5 * self.m * (self.w**2) * np.sum(x_vars**2 + (self.g*y_vars)**2)
        
        new_w = self.w * np.sqrt(initial_potential_energy/would_be_new_potential_energy)
        print(f"Initial w: {self.w}, final w: {new_w}")
        self.w = new_w
        
        
#         if conserve_energy
#         if conserve_energy:
#             new_potential_energy = self.get_energy()
#             desired_kinetic_energy = initial_potential_energy + initial_kinetic_energy - new_potential_energy
#             assert desired_kinetic_energy > 0, "Desired kinetic energy is negative!"
#             new_T = desired_kinetic_energy/(1 * self.N)
#             print(f"Temperature updated from {self.T} to {new_T}")
#             self.T = new_T

###########################################################################
###########################################################################
###########################################################################
# General functions
###########################################################################
###########################################################################
###########################################################################


def objective_function(vars, N, g, m = 1., w = 1., k = 1.):
    x_vars = vars[:N]  # First N variables are y_i
    y_vars = vars[N:]   # Next N variables are z_i
    
    # Compute the first term: 1/2 * (y_i^2 + (omega_z^2 / omega_y^2) * z_i^2)
    term1 = 0.5 * m * (w**2) * np.sum(x_vars**2 + (g*y_vars)**2)

    # Compute the second term: sum of 1/|r_i - r_j| for i < j
    term2 = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r_i = np.array([x_vars[i], y_vars[i]])  # Position of the i-th particle
            r_j = np.array([x_vars[j], y_vars[j]])  # Position of the j-th particle
            distance = np.linalg.norm(r_i - r_j)  # Euclidean distance
            if distance != 0:  # Avoid division by zero
                term2 += 1 / distance

    return term1 + term2*k

def fixed_point_equation(position_vectors_flat, g, w = 1., k = 1., m = 1.):
    assert position_vectors_flat.dtype == float, "The position vectors must have data type float"

    N = int(len(position_vectors_flat)/2)

    position_vectors = position_vectors_flat.reshape((N, 2))
    # Note that position_vectors_flat is of the form [x1,y1,x2,y2,x3,y3] NOT [x1,x2,x3,y1,y2,y3]
    # position_vectors is of the form [[x1,y1],[x2,y2],[x3,y3]]
        
    equations = np.zeros_like(position_vectors)

    # Diagonal anisotropy matrix
    Gamma = np.diag([1.0, g**2])
    
    for i in range(N):
        r_i = position_vectors[i]
        sum_term = np.zeros(2)

        for q in range(N):
            if q != i:
                r_q = position_vectors[q]
                diff = r_q - r_i
                norm_sq = np.dot(diff, diff)
                sum_term += (diff) / (norm_sq**(3/2))

        equations[i] = m * w**2 * (Gamma @ r_i) +k * sum_term
    
    return equations.flatten()

def positions_to_xy(positions):
    # Takes [[x1,y1],[x2,y2],[x3,y3]]
    # Returns [x1, x2, x3], [y1, y2, y3]
    return positions[:,0], positions[:,1]

def xy_to_positions(x, y):
    # Takes [x1, x2, x3], [y1, y2, y3]
    # Returns [[x1,y1],[x2,y2],[x3,y3]]
    return np.column_stack((x,y))

def positions_to_vars(positions):
    # Takes [[x1,y1],[x2,y2],[x3,y3]]
    # Returns [x1,x2,x3,y1,y2,y3]
    x, y = positions_to_xy(positions)
    
    return np.array(x.tolist() + y.tolist())

def vars_to_positions(vars):
    # Takes [x1,x2,x3,y1,y2,y3]
    # Returns [[x1,y1],[x2,y2],[x3,y3]]
    n = len(vars) // 2
    x = vars[:n]
    y = vars[n:]
    return xy_to_positions(x, y)

def vars_to_xy(vars):
    
    return positions_to_xy(vars_to_positions(vars))

def positions_to_flattened_positions(positions):
    # Takes [[x1,y1],[x2,y2],[x3,y3]]
    # Returns [x1, y1, x2, y2, x3, y3]
    return positions.flatten()        

def minimum_matching_distance(A, B):
    # Compute the pairwise distance matrix
    dists = np.linalg.norm(A[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)  # shape (N, N)
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(dists)
    
    # Compute the total matching distance
    min_distance = np.linalg.norm(A[row_ind] - B[col_ind])
    
    return min_distance

def base_radius(N):
    # Returns the radius r_star such that equally spaced points are in equilibrium at this radius
    
    if N <= 1:
        raise ValueError("N must be greater than 1.")
    
    summation = sum(1 / math.sin(math.pi * q / N) for q in range(1, N))
    r_star = (0.25 * summation) ** (1/3)
    return r_star

########################################################################################
########################################################################################
########################################################################################
# Quenching functions
########################################################################################
########################################################################################
########################################################################################

def generate_quench_animation(N, w, g, k, m, num_steps, loadfile, trajectory_folder, g_start, g_end, g_step, T = 0, dt = 1e-2):
    os.makedirs(trajectory_folder, exist_ok = True)
    
    S = Simulation(
        N = N,
        w = w,
        g = g,
        k = k,
        m = m,
        T = T,
        dt = dt,
        num_steps = num_steps
        )
    
    S.load_positions(loadfile)
    
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
        S.quench(gamma, True)
        print(f"Running gamma = {gamma}")
        S.run()
#       S.plot_density_map(save = True)
        S.save_trajectory(traj_file)
    
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
#         radius = base_radius(self.N)

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
