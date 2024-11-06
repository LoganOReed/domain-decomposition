import numpy as np
import matplotlib.pyplot as plt

def schwarz_domain_decomposition(f, L, N, overlap, overlap_center, max_iter=100, tol=1e-8, filename='test', visual=True):
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
    - errors, x1, u1, x2, u2
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
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.set_title('Solution on Overlapping Subdomains')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'output/{filename}.png')
        plt.close()

    return errors, x1, u1, x2, u2


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
    f=lambda x: np.where(x <= 0.5, np.sin(10 * np.pi * x), -1*np.sin(100 * np.pi * x))
    different_overlap_locations(f,
                                N = 5000,
                                overlap_size = 100,
                                overlap_centers = [0.3,0.5,0.7],
                                filenames = ['schwarz_left', 'schwarz_center', 'schwarz_right'],
                                visual = True)
    different_overlap_sizes(f,
                                N = 5000,
                                overlap_sizes = [100,500,1000],
                                overlap_center = 0.5,
                                filenames = ['schwarz_small', 'schwarz_medium', 'schwarz_large'])
