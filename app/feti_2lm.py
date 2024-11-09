# This is an equivalent algorithm to lions

# TODO:
# 1. Either find a new canonical f or hide the discontinuity
# 2. use this to recreate all of the lions code and have correct errors
# 3. Pick a couple of interesting hw problems

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Define the source function for the Poisson equation
# def f(x):
#     return np.ones_like(x)  # Example: constant source term

def f(x, x0=[0.2], width=0.1, period=100):
    return sum(np.exp(-((x - xi) / width)**2) * np.sin(period * np.pi * x) for xi in x0)

def f_continuous(x):
        # Sigmoid-like function to smooth the transition at x = 0.5
        # Smooth transition function with a sharp change in oscillation at x = 0.5
        smooth_transition = 1 / (1 + np.exp(-50 * (x - 0.5)))

        # Low-frequency oscillation part (low-frequency sine wave)
        low_oscillation = np.sin(2 * np.pi * x)

        # High-frequency oscillation part (high-frequency sine wave)
        high_oscillation = np.sin(2 * np.pi * 10 * x)

        # Blend the two oscillations smoothly using the smooth transition
        result = (1 - smooth_transition) * low_oscillation + smooth_transition * high_oscillation

        return result




def f_continuous(x):
    """Multiscale continuous function with a smooth transition at x = 0.5."""
    return np.where(x <= 0.5, -0.02 *np.sin(2 * np.pi * x),  np.sin(20 * np.pi * x))

def f_multiscale(x):
    """Multiscale source term with both low- and high-frequency components."""
    return np.sin(2 * np.pi * x) + 0.5 * np.sin(20 * np.pi * x) + 0.1 * np.sin(100 * np.pi * x)


# Function to construct a Poisson matrix for a given subdomain
def poisson_matrix(N, dx):
    """Constructs a sparse matrix for the 1D Poisson equation with Dirichlet BCs."""
    main_diag = -2.0 * np.ones(N)
    off_diag = np.ones(N - 1)
    diagonals = [main_diag, off_diag, off_diag]
    A = diags(diagonals, [0, -1, 1], format="csr") / dx**2
    return A

# FETI-2LM solver with adaptive omega and gamma
def feti_2lm_poisson(f, L, interface_location, N1, N2, initial_omega=0.5, initial_gamma=1.0, tol=1e-6, max_iter=1000):
    # Define subdomains and grids
    x1 = np.linspace(0, interface_location, N1 + 1)
    x2 = np.linspace(interface_location, L, N2 + 1)
    h1, h2 = x1[1] - x1[0], x2[1] - x2[0]
    
    # Create Poisson matrices for each subdomain
    A1 = poisson_matrix(N1 - 1, h1)
    A2 = poisson_matrix(N2 - 1, h2)
    f1 = f(x1[1:-1])  # Source term for interior points in subdomain 1
    f2 = f(x2[1:-1])  # Source term for interior points in subdomain 2

    # Initial guesses for solutions and Lagrange multipliers
    u1 = np.zeros(N1 + 1)  # Solution for subdomain 1
    u2 = np.zeros(N2 + 1)  # Solution for subdomain 2
    lambda_interface = 0.0  # Lagrange multiplier for interface continuity
    lambda_global = 0.0  # Second-level Lagrange multiplier for compatibility

    # Initialize omega and gamma
    omega, gamma = initial_omega, initial_gamma

    # Parameters for adaptive adjustment
    increase_factor = 1.1  # Factor to increase omega or gamma if convergence is steady
    decrease_factor = 0.9  # Factor to decrease omega or gamma if oscillations are detected
    max_omega, max_gamma = 1.0, 2.0  # Upper bounds for omega and gamma
    min_omega, min_gamma = 0.1, 0.1  # Lower bounds for omega and gamma

    # Convergence loop
    errors = []
    for iteration in range(max_iter):
        u1_old, u2_old = u1.copy(), u2.copy()
        
        # Step 1: Solve each subdomain with interface conditions
        # Subdomain 1 (left side)
        right_boundary_value = u2[1] - lambda_interface  # Interface condition with lambda
        f1_adj = f1.copy()
        f1_adj[-1] += (1 / h1) * right_boundary_value  # Adjust source term for the interface
        u1[1:-1] = spsolve(A1, f1_adj)  # Solve for interior of subdomain 1

        # Subdomain 2 (right side)
        left_boundary_value = u1[-2] + lambda_interface  # Interface condition with lambda
        f2_adj = f2.copy()
        f2_adj[0] += (1 / h2) * left_boundary_value  # Adjust source term for the interface
        u2[1:-1] = spsolve(A2, f2_adj)  # Solve for interior of subdomain 2

        # Step 2: Update Lagrange multiplier for interface continuity
        interface_jump = u1[-2] - u2[1]  # Difference between the two subdomain solutions at the interface
        lambda_interface += omega * interface_jump  # Update rule for the interface Lagrange multiplier

        # Step 3: Update global Lagrange multiplier for compatibility
        global_jump = (u1[-2] + u2[1]) / 2  # Average jump for global compatibility
        lambda_global += gamma * global_jump  # Update rule for the global multiplier

        # Step 4: Calculate error and check for convergence
        error = max(np.linalg.norm(u1 - u1_old), np.linalg.norm(u2 - u2_old))
        errors.append(error)
        
        # Adaptive adjustment of omega and gamma
        if len(errors) > 1:
            # If convergence is steady, increase omega and gamma up to their max values
            if errors[-1] < errors[-2]:
                omega = min(omega * increase_factor, max_omega)
                gamma = min(gamma * increase_factor, max_gamma)
            # If oscillations are detected (error increases), decrease omega and gamma
            else:
                omega = max(omega * decrease_factor, min_omega)
                gamma = max(gamma * decrease_factor, min_gamma)
        
        if error < tol:
            print(f"Converged after {iteration + 1} iterations with error = {error:.6e}")
            break
    else:
        print(f"Reached maximum iterations ({max_iter}) with final error = {error:.6e}")

    # Combine results for plotting
    x_combined = np.concatenate([x1[:-1], x2])
    u_combined = np.concatenate([u1[:-1], u2])

    # Plot the result
    plt.plot(x_combined, u_combined, label="FETI-2LM Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("1D Poisson Equation Solution with FETI-2LM (Adaptive)")
    plt.legend()
    plt.grid()
    plt.savefig('test.png')
    
    return u_combined, errors

# Parameters for subdomains and interface
L = 1.0
interface_location = 0.6
N1, N2 = 200, 5000

# Run the FETI-2LM solver with adaptive omega and gamma
# u_combined, errors = feti_2lm_poisson(f, L, interface_location, N1, N2, initial_omega=0.5, initial_gamma=0.3, tol=1e-12,max_iter=5000)
# u_combined, errors = feti_2lm_poisson(f_continuous, L, interface_location, N1, N2, initial_omega=0.5, initial_gamma=0.3, tol=1e-14,max_iter=2000)

# Define a grid of omega and gamma values
# Used to find optimal gamma omega
omega_grid = np.linspace(0.0010,0.2,4)
gamma_grid = np.linspace(0.0001,0.2,6)

best_omega, best_gamma = None, None
min_error = float('inf')

for omega in omega_grid:
    for gamma in gamma_grid:
        print(f"Testing omega = {omega}, gamma = {gamma}")
        u_combined, errors = feti_2lm_poisson(f_continuous, L, interface_location, N1, N2, initial_omega=omega, initial_gamma=gamma, tol=1e-14,max_iter=2000)
      
        # Final error for this combination
        final_error = errors[-1] if errors else float('inf')
      
        # Check if this combination is the best so far
        if final_error < min_error:
            min_error = final_error
            best_omega, best_gamma = omega, gamma
            print(f"New best omega, gamma = ({best_omega}, {best_gamma}) with error {min_error:.6e}")

print(f"Optimal values: omega = {best_omega}, gamma = {best_gamma} with final error = {min_error:.6e}")
