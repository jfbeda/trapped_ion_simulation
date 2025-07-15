# Written 2025-07-14

import math

def positions_to_xy(positions):
    # Takes [[x1,y1],[x2,y2],[x3,y3]]
    # Returns [x1, x2, x3], [y1, y2, y3]
    return positions[:,0], positions[:,1]

def xy_to_positions(x, y):
    # Takes [x1, x2, x3], [y1, y2, y3]
    # Returns [[x1,y1],[x2,y2],[x3,y3]]
    return np.column_stack((x,y))

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

   
def minimum_matching_distance(A, B):
    # Compute the pairwise distance matrix
    dists = np.linalg.norm(A[:, np.newaxis, :] - B[np.newaxis, :, :], axis=2)  # shape (N, N)
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(dists)
    
    # Compute the total matching distance
    min_distance = np.linalg.norm(A[row_ind] - B[col_ind])
    
    return min_distance

def fixed_point_equation(position_vectors_flat, g, w, k, m):
        """
        Returns the force-balance equations for a static configuration of N particles.

        Args:
            position_vectors_flat (np.ndarray): Flattened (N*2,) array of particle positions [x1, y1, x2, y2, ...].
            g (float): Isotropy parameter.
            w (float): Trap frequency.
            k (float): Coulomb constant (or scaling factor).
            m (float): Particle mass.

        Returns:
            np.ndarray: Flattened (N*2,) array of residual forces (should be ~0 at equilibrium).
        """
        
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

def base_radius(N, m, w, k):
    # Returns the radius r_star such that equally spaced points are in equilibrium at this radius
    
    if N <= 1:
        raise ValueError("N must be greater than 1.")
    
    summation = sum(1 / math.sin(math.pi * q / N) for q in range(1, N))
    r_star = (0.25 * summation/(m*(w**2)/k)) ** (1/3)
    return r_star
