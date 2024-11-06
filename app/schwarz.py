import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def f(x):
    return np.where(x <= 0.5, np.sin(10*np.pi * x), -1*np.sin(100*np.pi * x))

def f_boundary_layer(x, eps_inverse=100):
    return np.exp(-eps_inverse * x) * np.sin(eps_inverse * np.pi * x)

def f_interior_layer(x, x0=[0.3, 0.7], width=0.025, period=1):
    return sum(np.exp(-((x - xi) / width)**2) * np.sin(period * np.pi * x) for xi in x0)


def u_explicit(x):
    """explicit computation of solution"""
    return np.where(x <= 0.5, np.divide(np.sin(10*np.pi * x),10*10*np.pi*np.pi), np.divide(-1*np.sin(100*np.pi * x),100*100*np.pi*np.pi))
 

def schwarz_domain_decomposition(f, L, N, overlap, overlap_center, max_iter=4000, tol=1e-15, filename='test', visual=True):
    """
    Runs the Schwarz domain decomposition method with overlapping subdomains.

    Parameters:
    - f: function, source term function
    - L: float, length of the domain
    - N: int, total number of grid points in the full domain
    - overlap: int, number of overlapping grid points between subdomains
    - overlap_center: float, position (0 <= overlap_center <= L) at which the overlap is centered
    - max_iter: int, maximum number of iterations (default 1000)
    - tol: float, tolerance for convergence (default 1e-8)
    - filename: string, name for the png which contains the figures
    - visual: bool, true if the visuals should be created

    Returns:
    - errors, x1, x2, u1, u2
    """

    # Domain setup
    x = np.linspace(0, L, N + 1)
    h = L / N  # Grid spacing

    # Determine the boundary points for the two subdomains
    overlap_start = overlap_center - (overlap / 2) * h
    overlap_end = overlap_center + (overlap / 2) * h

    # Define the number of grid points in each subdomain
    N1 = int((overlap_start / L) * N) + overlap  # First subdomain with overlap
    N2 = int(((L - overlap_end) / L) * N) + overlap  # Second subdomain with overlap

    # Create the grid for each subdomain
    x1 = np.linspace(0, overlap_end, N1 + 1)
    x2 = np.linspace(overlap_start, L, N2 + 1)
    h1 = x1[1] - x1[0]
    h2 = x2[1] - x2[0]

    # Initial guess for the solution
    u1 = np.zeros_like(x1)
    u2 = np.zeros_like(x2)

    # Helper function for Schwarz iteration
    def schwarz_iteration(u1, u2, f, x1, x2, h1, h2, max_iter, tol):
        errors = []
        for k in range(max_iter):
            u1_old = u1.copy()
            u2_old = u2.copy()
            
            # Solve on the first subdomain
            for i in range(1, len(u1) - 1):
                u1[i] = 0.5 * (u1[i - 1] + u1[i + 1] - h1**2 * f(x1[i]))
            u1[-1] = np.interp(x1[-1], x2, u2)  # Update interface condition
            
            # Solve on the second subdomain
            for i in range(1, len(u2) - 1):
                u2[i] = 0.5 * (u2[i - 1] + u2[i + 1] - h2**2 * f(x2[i]))
            u2[0] = np.interp(x2[0], x1, u1)  # Update interface condition
            
            # Calculate the error (L2 norm of the difference)
            error = np.sqrt(np.sum((u1 - u1_old)**2) + np.sum((u2 - u2_old)**2))
            errors.append(error)
            
            # Check convergence
            if error < tol:
                break
                
        return u1, u2, errors

    # Run the Schwarz iteration
    u1, u2, errors = schwarz_iteration(u1, u2, f, x1, x2, h1, h2, max_iter, tol)

    if visual:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot the convergence rate on the first subplot
        ax1.plot(errors)
        ax1.set_yscale('log')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Error (L2 norm)')
        ax1.set_title('Convergence of Schwarz Domain Decomposition with Overlapping Subdomains')
        ax1.grid(True)

        # Plot the final solution on the second subplot
        ax2.plot(x1, u1, label='Subdomain 1')
        ax2.plot(x2, u2, label='Subdomain 2')
        ax2.plot(x1[-overlap:], 0.5*u1[-overlap:] + 0.5*u2[:overlap], label='Overlap')
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.set_title('Solution on Overlapping Subdomains')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'output/schwarz/{filename}.png')
        plt.close()

    return errors, x1, x2, u1, u2

def multi_schwarz_domain_decomposition(f, L, N, overlap_sizes, interface_points, max_iter=4000, tol=1e-15, filename='multi_test', visual=True):
    """
    Runs the Schwarz domain decomposition method with multiple overlapping subdomains, allowing variable overlap sizes.

    Parameters:
    - f: function, source term function
    - L: float, length of the domain
    - N: int, total number of grid points in the full domain
    - overlap_sizes: list of int, number of overlapping grid points between each pair of subdomains
    - interface_points: list of floats, positions of the interfaces between subdomains
    - max_iter: int, maximum number of iterations (default 4000)
    - tol: float, tolerance for convergence (default 1e-15)
    - filename: string, name for the png which contains the figures
    - visual: bool, true if the visuals should be created

    Returns:
    - errors: list of errors at each iteration
    - solutions: list of solutions for each subdomain
    - subdomain_grids: list of grids for each subdomain
    """

    # Define the complete domain grid
    x_full = np.linspace(0, L, N + 1)
    h = L / N  # Grid spacing

    # Define the boundaries of each subdomain based on the interface points and variable overlaps
    all_boundaries = [0] + interface_points + [L]
    subdomains = []
    solutions = []
    overlaps = []  # To store overlap regions for plotting

    for i in range(len(all_boundaries) - 1):
        # Calculate overlap sizes for the left and right boundaries of the subdomain
        overlap_left = overlap_sizes[i - 1] if i > 0 else 0  # No left overlap for the first subdomain
        overlap_right = overlap_sizes[i] if i < len(overlap_sizes) else 0  # No right overlap for the last subdomain

        # Define grid points in each subdomain with specified overlaps
        x_start = max(0, all_boundaries[i] - overlap_left * h)
        x_end = min(L, all_boundaries[i + 1] + overlap_right * h)
        num_points = int((x_end - x_start) / h)
        x_sub = np.linspace(x_start, x_end, num_points + 1)
        subdomains.append(x_sub)
        solutions.append(np.zeros_like(x_sub))

        # Store overlap regions for plotting
        if overlap_left > 0 and i > 0:
            overlap_start = all_boundaries[i] - overlap_left * h
            overlap_end = all_boundaries[i]
            overlaps.append((overlap_start, overlap_end))
        if overlap_right > 0 and i < len(all_boundaries) - 2:
            overlap_start = all_boundaries[i + 1]
            overlap_end = all_boundaries[i + 1] + overlap_right * h
            overlaps.append((overlap_start, overlap_end))

    # Helper function for multi-subdomain Schwarz iteration
    def multi_schwarz_iteration(f, subdomains, solutions, h, overlap_sizes, max_iter, tol):
        errors = []
        for k in range(max_iter):
            old_solutions = [u.copy() for u in solutions]
            
            # Update each subdomain
            for i in range(len(subdomains)):
                x_sub = subdomains[i]
                u_sub = solutions[i]
                f_values = f(x_sub)

                # Solve the interior points of the subdomain
                for j in range(1, len(u_sub) - 1):
                    u_sub[j] = 0.5 * (u_sub[j - 1] + u_sub[j + 1] - h**2 * f_values[j])
                
                # Update the overlapping interfaces with neighboring subdomains
                if i > 0:  # Has a left neighbor
                    u_left = solutions[i - 1]
                    overlap_idx_left = int(overlap_sizes[i - 1])  # Size of overlap with the left neighbor
                    u_sub[0] = u_left[-overlap_idx_left]  # Use rightmost overlap from the left neighbor
                if i < len(subdomains) - 1:  # Has a right neighbor
                    u_right = solutions[i + 1]
                    overlap_idx_right = int(overlap_sizes[i])  # Size of overlap with the right neighbor
                    u_sub[-1] = u_right[overlap_idx_right]  # Use leftmost overlap from the right neighbor
            
            # Compute the error
            error = sum(np.linalg.norm(u - u_old) for u, u_old in zip(solutions, old_solutions))
            errors.append(error)
            if error < tol:
                break
        
        return solutions, errors

    # Run the multi-subdomain Schwarz iteration
    solutions, errors = multi_schwarz_iteration(f, subdomains, solutions, h, overlap_sizes, max_iter, tol)

    # Visualize results if specified
    if visual:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot the convergence rate on the first subplot
        ax1.plot(errors)
        ax1.set_yscale('log')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Error (L2 norm)')
        ax1.set_title('Convergence of Multi-Subdomain Schwarz Domain Decomposition')
        ax1.grid(True)

        # Plot the final solution on the second subplot
        for i, x_sub in enumerate(subdomains):
            ax2.plot(x_sub, solutions[i], label=f'Subdomain {i+1}')
        
        # Plot each overlap as a separate region
        for j, (overlap_start, overlap_end) in enumerate(overlaps):
            ax2.axvspan(overlap_start, overlap_end, color='gray', alpha=0.3, label=f'Overlap' if j == 0 else None)

        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.set_title('Solution on Multiple Overlapping Subdomains')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'output/schwarz/{filename}.png')
        plt.close()

    return errors, subdomains, solutions




def different_overlap_locations(f,N,overlap_size,overlap_centers,filenames, visual=False):
    """A couple of different overlap locations"""
    errors = []
    for i in range(len(overlap_centers)):
        errors_tmp, _, _, _, _ = schwarz_domain_decomposition(
            f=f,
            L=1.0,
            N=N,
            overlap=overlap_size,
            overlap_center=overlap_centers[i],
            filename=filenames[i],
            visual=visual
        )
        errors.append(errors_tmp)

    for i in range(len(overlap_centers)):
        plt.plot(errors[i], label=f'Overlap centered at {overlap_centers[i]}')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (L2 norm)')
    plt.title('Convergence of Schwarz Domain Decomposition with Overlapping Subdomains')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'output/compare/schwarz_convergence_overlap_location.png')
    plt.close()

def different_overlap_sizes(f,N,overlap_sizes,overlap_center,filenames,visual=False):
    """A couple of different overlap locations"""
    errors = []
    for i in range(len(overlap_sizes)):
        errors_tmp, _, _, _, _ = schwarz_domain_decomposition(
            f=f,
            L=1.0,
            N=N,
            overlap=overlap_sizes[i],
            overlap_center=overlap_center,
            filename=filenames[i],
            visual=visual
        )
        errors.append(errors_tmp)

    for i in range(len(overlap_sizes)):
        plt.plot(errors[i], label=f'Overlap size: {overlap_sizes[i]}')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (L2 norm)')
    plt.title('Convergence of Schwarz Domain Decomposition with Overlapping Subdomains')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'output/compare/schwarz_convergence_overlap_size.png')
    plt.close()

if __name__ == "__main__":

    # schwarz_domain_decomposition(f_boundary_layer,L=1.0, N=5000, overlap=100, overlap_center=0.15, max_iter=1000, tol=1e-15, filename='boundary_layer')
    # schwarz_domain_decomposition(partial(f_interior_layer, x0=[0.5], period = 100),L=1.0, N=5000, overlap=1500, overlap_center=0.5, max_iter=1000, tol=1e-15, filename='interior_layer')
    # different_overlap_locations(f,
    #                             N = 5000,
    #                             overlap_size = 200,
    #                             overlap_centers = [0.3,0.5,0.7],
    #                             filenames = ['left', 'center', 'right'],
    #                             visual = True)
    # different_overlap_sizes(f,
    #                             N = 5000,
    #                             overlap_sizes = [100,500,1000],
    #                             overlap_center = 0.5,
    #                             filenames = ['small', 'medium', 'large'])
    #
    multi_schwarz_domain_decomposition(f_interior_layer, 1.0, 5000, [200, 50, 150], [0.25, 0.5, 0.85], filename='interior_layers_multiple_subdomain', visual=True)

