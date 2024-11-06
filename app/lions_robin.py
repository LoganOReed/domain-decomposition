import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.where(x <= 0.5, np.sin(10*np.pi * x), -1*np.sin(100*np.pi * x))

def f_boundary_layer(x, eps_inverse=100):
    return np.exp(-eps_inverse * x) * np.sin(eps_inverse * np.pi * x)

def g(x):
    return np.sin((10*np.pi)/(x + 0.0001))

def lions_robin_domain_decomposition(f, L, interface_point, N1, N2, eta=1.0, alpha_robin=0.3, tol=1e-18, max_iter=5000, filename='temp', visual=True):
    """
    Solves the 1D Poisson equation using P.-L. Lions' domain decomposition method with Robin transmission conditions.
    
    Parameters:
    f: function
        The source term as a vectorized Python function.
    L: float
        Length of the domain.
    interface_point: float
        The point where the two subdomains meet.
    N1: int
        Number of grid points in the first subdomain.
    N2: int
        Number of grid points in the second subdomain.
    eta: float, optional
        Parameter for the differential equation (default is 1.0).
    alpha_robin: float, optional
        Coefficient for the Robin boundary condition (default is 1.0).
    tol: float, optional
        Tolerance for convergence (default is 1e-6).
    max_iter: int, optional
        Maximum number of iterations (default is 1000).
    visual: bool, optional
        Determines if plots are constructed (default is True)
    
    Returns:
    u1: numpy array
        Approximate solution in the first subdomain.
    u2: numpy array
        Approximate solution in the second subdomain.
    errors: list
        List of errors at each iteration.
    """
    # Subdomain setups with different grid spacings
    x1 = np.linspace(0, interface_point, N1 + 1)
    x2 = np.linspace(interface_point, L, N2 + 1)
    h1 = x1[1] - x1[0]  # Grid spacing for the first subdomain
    h2 = x2[1] - x2[0]  # Grid spacing for the second subdomain

    # Initial guess for lambda values on the interface
    lambda1 = 0.0
    lambda2 = 0.0

    # Initial guess for the solution
    u0_1 = np.zeros_like(x1)
    u0_2 = np.zeros_like(x2)

    # Helper function to solve the subdomain problem with Robin boundary condition
    def solve_subdomain(f_values, x_sub, h, lambda_value, alpha_robin):
        # Create a new array for the subdomain solution
        u_new = np.zeros_like(x_sub)
        
        # Number of points in the subdomain
        N_sub = len(x_sub) - 1

        # Solve the interior points of the subdomain
        for i in range(1, N_sub):
            u_new[i] = (u_new[i - 1] + u_new[i + 1] - h**2 * f_values[i]) / (2 + h**2 * eta)
        
        # Apply Robin boundary condition at the interface point of the subdomain
        u_new[-1] = (lambda_value + alpha_robin * u_new[-2]) / (alpha_robin + 1 / h)
        
        return u_new

    # Main function implementing P.-L. Lions' algorithm with Robin boundary conditions
    def lions_domain_decomposition_robin(f, u0_1, u0_2, x1, x2, h1, h2, alpha_robin, tol, max_iter):
        u1, u2 = u0_1.copy(), u0_2.copy()
        errors = []
        
        lambda1, lambda2 = 0.0, 0.0  # Initial lambda values
        
        for iter in range(max_iter):
            u1_old, u2_old = u1.copy(), u2.copy()
            
            # Solve on the first subdomain with lambda1 as the boundary condition
            f_values1 = f(x1)
            u1 = solve_subdomain(f_values1, x1, h1, lambda1, alpha_robin)
            
            # Solve on the second subdomain with lambda2 as the boundary condition
            f_values2 = f(x2)
            u2 = solve_subdomain(f_values2, x2, h2, lambda2, alpha_robin)
            
            # Update lambda values according to the transmission conditions
            lambda1_new = -lambda2 + 2 * alpha_robin * u2[0]
            lambda2_new = -lambda1 + 2 * alpha_robin * u1[-1]
            
            # Calculate error and check convergence
            error = np.linalg.norm(np.concatenate((u1, u2)) - np.concatenate((u1_old, u2_old)), ord=2)
            errors.append(error)
            
            if error < tol:
                break
            
            # Update lambdas for the next iteration
            lambda1, lambda2 = lambda1_new, lambda2_new
        
        return u1, u2, errors

    # Run the domain decomposition method
    u1, u2, errors = lions_domain_decomposition_robin(f, u0_1, u0_2, x1, x2, h1, h2, alpha_robin, tol, max_iter)

    # Combine the solutions and grids for plotting
    # Remove duplicate interface point if necessary
    if x1[-1] == x2[0]:
        x_combined = np.concatenate((x1, x2[1:]))
        u_combined = np.concatenate((u1, u2[1:]))
    else:
        x_combined = np.concatenate((x1, x2))
        u_combined = np.concatenate((u1, u2))

    if visual:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot the convergence rate on the first subplot
        ax1.plot(errors,label=f'alpha = {alpha_robin}')
        ax1.set_yscale('log')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Error (L2 norm)')
        ax1.set_title("Convergence of P.-L. Lions' Domain Decomposition Method")
        ax1.legend()
        ax1.grid(True)

        # Plot the final solution on the second subplot
        ax2.plot(x1, u1, label=f'Ω_1 = (0, {interface_point}) with N={N1}')
        ax2.plot(x2, u2, label=f'Ω_2 = ({interface_point}, {L}) with N={N2}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('u(x)')
        ax2.set_title("Solution of the PDE using P.-L. Lions' Domain Decomposition")
        ax2.legend()
        ax2.grid(True)

        # Show the combined plot
        plt.tight_layout()
        plt.savefig(f'output/lions/{filename}.png')
        plt.close()

    return x1, x2, u1, u2, errors



def different_interface_points(f,interface_points,N1s,N2s,filenames,L=1.0,visual=False):
    """docstring for different_interface_points"""
    errors = []
    x1s = []
    x2s = []
    u1s = []
    u2s = []
    for i in range(len(interface_points)):
        x1_tmp, x2_tmp, u1_tmp, u2_tmp, error_tmp = lions_robin_domain_decomposition(f, L, interface_points[i], N1s[i], N2s[i], alpha_robin=0.6, filename=filenames[i], visual=visual)
        x1s.append(x1_tmp)
        x2s.append(x2_tmp)
        u1s.append(u1_tmp)
        u2s.append(u2_tmp)
        errors.append(error_tmp)


    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('(Non-Overlapping) P.L. Lions Interface Location Comparison')


    for i in range(len(interface_points)):
        axs[0,0].plot(errors[i], label=f'Interface Point: {interface_points[i]}')
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlabel('Iteration')
    axs[0,0].set_ylabel('Error (L2 norm)')
    axs[0,0].legend()
    axs[0,0].grid(True)

    for i in range(1,len(interface_points)+1):
        axs[i // 2,i%2].plot(x1s[i-1], u1s[i-1], label=f'Ω_1 = (0, {interface_points[i-1]}) with N={N1s[i-1]}')
        axs[i // 2,i%2].plot(x2s[i-1], u2s[i-1], label=f'Ω_2 = ({interface_points[i-1]}, {L}) with N={N2s[i-1]}')
        axs[i // 2,i%2].set_xlabel('x')
        axs[i // 2,i%2].set_ylabel('u(x)')
        axs[i // 2,i%2].set_title(f'Inner Boundary: x = {interface_points[i-1]}')
        # ax2.set_title("Solution of the PDE using P.-L. Lions' Domain Decomposition")
        axs[i // 2,i%2].legend()
        axs[i // 2,i%2].grid(True)

    plt.savefig(f'output/compare/nonoverlap_interface_locations.png')
    plt.close()

def different_robin_alphas(f,interface_point,alpha_robins,N1,N2,filenames,L=1.0,visual=False):
    """docstring for different_interface_points"""
    errors = []
    x1s = []
    x2s = []
    u1s = []
    u2s = []
    for i in range(len(alpha_robins)):
        x1_tmp, x2_tmp, u1_tmp, u2_tmp, error_tmp = lions_robin_domain_decomposition(f, L, interface_point, N1, N2, alpha_robin=alpha_robins[i], filename=filenames[i], visual=visual)
        x1s.append(x1_tmp)
        x2s.append(x2_tmp)
        u1s.append(u1_tmp)
        u2s.append(u2_tmp)
        errors.append(error_tmp)

    for i in range(len(alpha_robins)):
        plt.plot(errors[i], label=f'Alpha: {alpha_robins[i]}')
    plt.yscale('log')
    plt.title('(Non-Overlapping) P.L. Lions Alpha Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Error (L2 norm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'output/compare/nonoverlap_robin_alphas.png')
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('(Non-Overlapping) P.L. Lions Alpha Convergence Comparison')

    for i in range(len(alpha_robins)):
        axs[0,0].plot(errors[i], label=f'Alpha: {alpha_robins[i]}')
    axs[0,0].set_yscale('log')
    axs[0,0].set_xlabel('Iteration')
    axs[0,0].set_ylabel('Error (L2 norm)')
    axs[0,0].legend()
    axs[0,0].grid(True)

    for i in range(1,len(alpha_robins)+1):
        axs[i // 2,i%2].plot(x1s[i-1], u1s[i-1], label=f'Ω_1 = (0, {interface_point}) with N={N1}')
        axs[i // 2,i%2].plot(x2s[i-1], u2s[i-1], label=f'Ω_2 = ({interface_point}, {L}) with N={N2}')
        axs[i // 2,i%2].set_xlabel('x')
        axs[i // 2,i%2].set_ylabel('u(x)')
        axs[i // 2,i%2].set_title(f'Alpha: {alpha_robins[i-1]}')
        # ax2.set_title("Solution of the PDE using P.-L. Lions' Domain Decomposition")
        axs[i // 2,i%2].legend()
        axs[i // 2,i%2].grid(True)

    plt.savefig(f'output/compare/nonoverlap_robin_alphas_with_solutions.png')
    plt.close()

def different_interface_points_and_robin_alphas(f,interface_points,alpha_robins,N1s,N2s,filenames,tol=1e-18,L=1.0,visual=False):
    """docstring for different_interface_points_and_robin_alphas"""

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle('(Non-Overlapping) P.L. Lions Alpha and Interface Comparison')

    for i in range(len(interface_points)):
        errors = []
        for j in range(len(alpha_robins)):
            _, _, _, _, error_tmp = lions_robin_domain_decomposition(f, L, interface_points[i], N1s[i], N2s[i], alpha_robin=alpha_robins[j],tol=tol, filename=filenames[i], visual=visual)
            errors.append(error_tmp)

        for j in range(len(alpha_robins)):
            axs[i // 2,i%2].plot(errors[j], label=f'Alpha: {alpha_robins[j]}')
        # axs[i // 2,i%2].set_ylim(10**(-14), 10**(-4))
        axs[i // 2,i%2].set_yscale('log')
        axs[i // 2,i%2].set_xlabel('Iteration')
        axs[i // 2,i%2].set_ylabel('Error (L2 norm)')
        axs[i // 2,i%2].set_title(f'Interface: x = {interface_points[i]}')
        axs[i // 2,i%2].legend()
        axs[i // 2,i%2].grid(True)
    plt.savefig(f'output/compare/nonoverlap_robin_alphas_and_interface_points.png')
    plt.close()

if __name__ == "__main__":
    # Length of the domain
    L = 1.0                
    # different_interface_points(f,[0.23,0.5,0.83], [100,100,1000], [1000,1000,100],filenames = ['nonoverlap_left', 'nonoverlap_center', 'nonoverlap_right'], visual=True)
    # different_interface_points(f,[0.23,0.5,0.83], [100,100,1000], [1000,1000,100],filenames = ['nonoverlap_left', 'nonoverlap_center', 'nonoverlap_right'], visual=False)
    # different_robin_alphas(f,0.83,[0.1,0.5,0.9], 1000, 100, filenames = ['nonoverlap_center_small_alpha', 'nonoverlap_center_medium_alpha', 'nonoverlap_center_large_alpha'], visual=False)
    # different_interface_points_and_robin_alphas(f,[0.3,0.5,0.7,0.9],[0.4,1.0,4.0], [100,100,1000,1000], [1000,1000,100,100], tol=1e-23, filenames = ['nonoverlap_left_large_alpha', 'nonoverlap_center_large_alpha', 'nonoverlap_right_large_alpha', 'nonoverlap_far_right_large_alpha'], visual=True)


    _,_,_,_, errors = lions_robin_domain_decomposition(f_boundary_layer, L, 0.05, 100, 100, alpha_robin=3.0, filename='boundary_layer', tol=1e-20)
    # u1, u2, errors = lions_robin_domain_decomposition(g, L, 0.2, 100, 160, alpha_robin=0.4, filename='nonlipschitz')
    # u1, u2, errors = lions_robin_domain_decomposition(f, L, 0.23, N1, N2, alpha_robin=0.6, filename='nonoverlap_left')
    # u1, u2, errors = lions_robin_domain_decomposition(f, L, 0.83, 1000, 100, alpha_robin=0.6, filename='nonoverlap_right')
