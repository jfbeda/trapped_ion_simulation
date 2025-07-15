import numpy as np
import natural_units as units 
import matplotlib.pyplot as plt
import numba
import matplotlib.animation as animation
import numpy as np
import natural_units as units 
import matplotlib.pyplot as plt
import numba
import matplotlib.animation as animation


class Laser:
    def __init__(self, wavelength_nm, direction, saturation, detuning, Gamma):
        self.wavelength = wavelength_nm / 1000.  # Convert nm to μm
        self.k_mag = 2 * np.pi / self.wavelength  # |k| in um⁻¹
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))  # Normalize
        self.k_vec = self.k_mag * self.direction
        self.saturation = saturation
        self.detuning = detuning  # In MHz (Δ ≈ -Γ/2 optimally)
        self.Gamma = Gamma        # In MHz (Γ)
        self.damping = (- units.hbar * self.k_mag**2 * (4 * self.saturation * self.detuning / self.Gamma) / (1 + self.saturation + (2 * self.detuning / self.Gamma)**2)**2)
        scatter_rate_0 = 0.5 * self.Gamma * self.saturation / (1 + self.saturation + (2 * self.detuning / self.Gamma)**2)
        self.force_at_zero = units.hbar * scatter_rate_0 * self.k_vec
        self.get_force(np.array([[0,0]])) # Call get_force to compile the numba function
        

    def get_force(self, velocities):
        return compute_laser_force_numba(
            velocities,
            self.k_vec,
            self.saturation,
            self.detuning,
            self.Gamma,
            units.hbar
        )
    
    
    def plot_force_vs_velocity(self, v_min=None, v_max=None, num_points=500, external_velocities=None):
        """
        Plot the Doppler cooling force as a function of velocity along the laser direction,
        including a linear approximation around v = 0 using the damping coefficient.
        """
        plt.figure(figsize=(8, 5))

        # Handle external velocities
        if external_velocities is not None:
            v_proj = np.dot(external_velocities, self.direction)
            forces = self.get_force(external_velocities)
            f_proj = np.dot(forces, self.direction)
            plt.scatter(v_proj, f_proj, label='Particle forces')

            if v_min is None:
                v_min = min(v_proj) * 1.1
            if v_max is None:
                v_max = max(v_proj) * 1.1
        else:
            if v_min is None:
                v_min = -100
            if v_max is None:
                v_max = 100

        # Theoretical force curve
        v_parallel = np.linspace(v_min, v_max, num_points)
        velocities = np.outer(v_parallel, self.direction)
        forces = self.get_force(velocities)
        force_along_k = np.dot(forces, self.direction)

        plt.plot(v_parallel, force_along_k, label='Full Doppler force', lw=2)

        # Linear approximation using damping
        linear_approx = np.dot(self.force_at_zero, self.direction) - self.damping * v_parallel
        plt.plot(v_parallel, linear_approx, '--', label='Linear approximation', lw=2)

        plt.xlabel('Velocity parallel to laser direction (μm/μs)')
        plt.ylabel('Force along k (amu μm/μs²)')
        plt.title('Laser Cooling Force vs Velocity')
        plt.grid(True)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.legend()
        plt.show()

    def animate_force_vs_velocity(self, all_velocities, v_min=None, v_max=None, num_points=500, interval=50, save_path=None):
        """
        Animate the Doppler cooling force with particles' velocities evolving over time.

        Parameters:
        - all_velocities: (T, N, 2) array of velocities for N particles over T timesteps.
        - v_min, v_max: (optional) range for plotting theoretical force curve; auto-scaled if not provided.
        - num_points: number of samples in theoretical force curve.
        - interval: delay between frames in milliseconds.
        - save_path: optional filepath to save animation (e.g., 'anim.mp4' or 'anim.gif').
        """

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        T = all_velocities.shape[0]

        # Compute all projections along laser direction
        v_proj_all = np.dot(all_velocities.reshape(-1, 2), self.direction)

        # Auto-scale v_min, v_max if not provided
        if v_min is None:
            v_min = v_proj_all.min() - 0.1 * (v_proj_all.max() - v_proj_all.min())
        if v_max is None:
            v_max = v_proj_all.max() + 0.1 * (v_proj_all.max() - v_proj_all.min())

        # Theoretical force curve
        v_parallel = np.linspace(v_min, v_max, num_points)
        velocity_grid = np.outer(v_parallel, self.direction)
        force_grid = self.get_force(velocity_grid)
        force_along_k = np.dot(force_grid, self.direction)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(v_parallel, force_along_k, label='Theoretical force curve', lw=2)
        scatter = ax.scatter([], [], color='red', label='Particle projections')

        ax.set_xlim(v_min, v_max)
        y_min = force_along_k.min()
        y_max = force_along_k.max()
        buffer = 0.1 * abs(y_max - y_min)
        ax.set_ylim(y_min - buffer, y_max + buffer)
        ax.set_xlabel('Velocity parallel to laser direction (μm/μs)')
        ax.set_ylabel('Force along k (amu μm/μs²)')
        ax.set_title('Laser Cooling Force vs Velocity')
        ax.grid(True)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend()

        def init():
            # Use dummy point so it's not empty
            scatter.set_offsets(np.array([[0, 0]]))
            return scatter,

        def update(frame):
            velocities_t = all_velocities[frame]  # shape (N, 2)
            forces_t = self.get_force(velocities_t)

            v_proj = np.dot(velocities_t, self.direction)  # shape (N,)
            f_proj = np.dot(forces_t, self.direction)       # shape (N,)
            points = np.column_stack((v_proj, f_proj))      # shape (N, 2)

            scatter.set_offsets(points)
            return scatter,

        ani = animation.FuncAnimation(
            fig, update, frames=T, init_func=init, interval=interval, blit=False, repeat=True
        )

        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=1000 // interval)
            else:
                ani.save(save_path, writer='ffmpeg', fps=1000 // interval)
            print(f"Animation saved to {save_path}")

        plt.close(fig)

#         return ani


        
##################################################################
        
@numba.njit#(fastmath = True)
def compute_laser_force_numba(velocities, k_vec, saturation, detuning, Gamma, hbar):
    N = velocities.shape[0]
    forces = np.zeros_like(velocities)

    for i in range(N):
        v_dot_k = 0.0
        for d in range(2):
            v_dot_k += velocities[i, d] * k_vec[d]

        delta_eff = detuning - v_dot_k
        denom = 1.0 + saturation + (2.0 * delta_eff / Gamma) ** 2
        scatter_rate = 0.5 * Gamma * saturation / denom

        for d in range(2):
            forces[i, d] = hbar * scatter_rate * k_vec[d]

    return forces

