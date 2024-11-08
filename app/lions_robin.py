import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def f(x):
    return np.where(x <= 0.5, np.sin(10*np.pi * x), -1*np.sin(100*np.pi * x))

def f_boundary_layer(x, eps_inverse=100):
    return np.exp(-eps_inverse * x) * np.sin(eps_inverse * np.pi * x)


def f_interior_layer(x, x0=[0.3, 0.7], width=0.025, period=1):
    return sum(np.exp(-((x - xi) / width)**2) * np.sin(period * np.pi * x) for xi in x0)

# def f_not_square_summable(x, eps_inverse=100):
#     return np.sin((10*np.pi) / x+ 0.0000000000001) / np.sqrt(x + 0.0000000000001)

def f_not_square_summable(x, eps_inverse=100):
    return 1 / np.sqrt(x + 0.00000000000001)


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
        
        lambda1, lambda2 = 1.0, 1.0  # Initial lambda values
        
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


def multi_subdomain_decomposition(f, L, interface_points, Ns, eta=1.0, alpha_robin=1.0, tol=1e-15, max_iter=1000, filename='test'):
    """
    Solves the 1D Poisson equation using P.-L. Lions' domain decomposition method with Robin transmission conditions
    across multiple subdomains.

    Parameters:
    - f: function
        The source term as a vectorized Python function.
    - L: float
        Length of the domain.
    - Ns: list of int
        Number of grid points for each subdomain.
    - interface_points: list of float
        Positions of the interfaces between subdomains.
    - eta: float, optional
        Parameter for the differential equation (default is 1.0).
    - alpha_robin: float, optional
        Coefficient for the Robin boundary condition (default is 0.3).
    - tol: float, optional
        Tolerance for convergence (default is 1e-8).
    - max_iter: int, optional
        Maximum number of iterations (default is 1000).

    Returns:
    - solutions: list of numpy arrays
        The approximate solutions in each subdomain.
    - errors: list
        List of errors at each iteration.
    """

    # Generate subdomain grids
    subdomains = []
    h_values = []
    all_x = [0] + interface_points + [L]
    
    for i in range(len(all_x) - 1):
        x_sub = np.linspace(all_x[i], all_x[i+1], int(Ns[i] + 1))
        subdomains.append(x_sub)
        h_values.append(x_sub[1] - x_sub[0])

    # Initial guesses for solutions and lambdas at each interface
    solutions = [np.zeros_like(x) for x in subdomains]
    lambdas = [0.0] * (len(interface_points) * 2)

    # Helper function to solve a subdomain problem with Robin boundary condition

    def solve_subdomain(f_values, x_sub, h, boundary_value, alpha_robin, eta=1.0):
        """
        Solves the Poisson equation in a single subdomain with Robin boundary coupling.

        Parameters:
        - f_values: array-like
            The source term evaluated at the grid points of the subdomain.
        - x_sub: array-like
            The grid points for the current subdomain.
        - h: float
            Grid spacing for the current subdomain.
        - boundary_value: float
            Value from the neighboring subdomain used in the Robin boundary condition.
        - alpha_robin: float
            Coefficient for the Robin boundary condition.
        - eta: float, optional
            Coefficient for the differential equation.

        Returns:
        - u_new: array-like
            Solution in the current subdomain after applying boundary conditions.
        """
        # Initialize the solution array
        u_new = np.zeros_like(x_sub)
        N_sub = len(x_sub) - 1  # Number of interior points

        # Iteratively solve the Poisson equation for the interior points
        for i in range(1, N_sub):
            u_new[i] = (u_new[i - 1] + u_new[i + 1] - h**2 * f_values[i]) / (2 + h**2 * eta)

        # Apply Robin boundary condition at the interface
        u_new[-1] = (alpha_robin * boundary_value + u_new[-2]) / (alpha_robin + 1 / h)

        return u_new


    # Main iterative solver for the multi-subdomain decomposition

    errors = []
    num_interfaces = len(interface_points)
    for iteration in range(max_iter):
        # Store a copy of the old solutions to track convergence
        old_solutions = [u.copy() for u in solutions]

        # Iterate over each subdomain
        for i, (x_sub, h, solution) in enumerate(zip(subdomains, h_values, solutions)):
            # Calculate the source term values for the current subdomain
            f_values = f(x_sub)

            # Apply boundary conditions based on neighboring subdomains
            if i == 0:
                # Left-most subdomain: Use the right boundary from the next subdomain
                boundary_value = solutions[i + 1][1] if num_interfaces > 0 else 0.0
                solutions[i] = solve_subdomain(f_values, x_sub, h, boundary_value, alpha_robin)

            elif i == len(subdomains) - 1:
                # Right-most subdomain: Use the left boundary from the previous subdomain
                boundary_value = solutions[i - 1][-2] if num_interfaces > 0 else 0.0
                solutions[i] = solve_subdomain(f_values, x_sub, h, boundary_value, alpha_robin)

            else:
                # Middle subdomains: Use both left and right boundaries from neighbors
                boundary_left = solutions[i - 1][-2]
                boundary_right = solutions[i + 1][1]
                # First apply the left boundary condition
                solutions[i] = solve_subdomain(f_values, x_sub, h, boundary_left, alpha_robin)
                # Then apply the right boundary condition directly using `boundary_right`
                solutions[i][-1] = (alpha_robin * boundary_right + solutions[i][-2]) / (alpha_robin + 1 / h)

        # Compute the error based on the difference in solutions
        error = sum(np.linalg.norm(u - u_old) for u, u_old in zip(solutions, old_solutions))
        errors.append(error)

        # Check for convergence
        if error < tol:
            print(f"Converged after {iteration + 1} iterations with error = {error:.6e}")
            break

# If max iterations reached without convergence
        else:
            print(f"Reached maximum iterations ({max_iter}) with final error = {error:.6e}")


    # Plot the combined results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot the convergence rate
    ax1.plot(errors,label=f'alpha = {alpha_robin}')
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error (L2 norm)')
    ax1.set_title("Convergence of P.-L. Lions' Domain Decomposition Method")
    ax1.legend()
    ax1.grid(True)



    # Plot the final solution
    for i, (x_sub, u_sub) in enumerate(zip(subdomains, solutions)):
        ax2.plot(x_sub, u_sub, label=f'Subdomain {i+1}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.set_title("Solution of the PDE using Multi-Subdomain Domain Decomposition")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'output/lions/{filename}.png')
    plt.close()

    return solutions, errors

if __name__ == "__main__":
    # Length of the domain
    L = 1.0                
    # different_interface_points(f,[0.23,0.5,0.83], [100,100,1000], [1000,1000,100],filenames = ['nonoverlap_left', 'nonoverlap_center', 'nonoverlap_right'], visual=True)
    # different_interface_points(f,[0.23,0.5,0.83], [100,100,1000], [1000,1000,100],filenames = ['nonoverlap_left', 'nonoverlap_center', 'nonoverlap_right'], visual=False)
    # different_robin_alphas(f,0.83,[0.1,0.5,0.9], 1000, 100, filenames = ['nonoverlap_center_small_alpha', 'nonoverlap_center_medium_alpha', 'nonoverlap_center_large_alpha'], visual=False)
    # different_interface_points_and_robin_alphas(f,[0.3,0.5,0.7,0.9],[0.4,1.0,4.0], [100,100,1000,1000], [1000,1000,100,100], tol=1e-23, filenames = ['nonoverlap_left_large_alpha', 'nonoverlap_center_large_alpha', 'nonoverlap_right_large_alpha', 'nonoverlap_far_right_large_alpha'], visual=True)


    # Shows boundary Layer
    # _,_,_,_, errors = lions_robin_domain_decomposition(f_boundary_layer, L, 0.05, 100, 100, alpha_robin=3.0, filename='boundary_layer', tol=1e-20)

    # TODO: Show figure with difference from actual solution, since smaller steps doesn't show anything when not convergent
    # _,_,_,_, errors = lions_robin_domain_decomposition(f_not_square_summable, L, 0.15, 1000, 1000, alpha_robin=10.5, filename='not_square_summable', tol=1e-32)

    # TODO: Put this and next (commented) setup into one plot to show difference in boundary choice
    # Since the latter's bdy is too close to the inner layer it causes the outer two subdomains to oscillate
    # _, errors = multi_subdomain_decomposition(partial(f_interior_layer, x0=[0.5], period=100), L, [0.4,0.6], [100, 1000, 100], alpha_robin=1.0, filename='interior_layer_good_boundary', tol=1e-25)
    # _, errors = multi_subdomain_decomposition(partial(f_interior_layer, x0=[0.5], period = 100), L, [0.45,0.55], [100, 1000, 100], alpha_robin=1.0, filename='interior_layer_bad_boundary', tol=1e-20)


    _, errors = lions_robin_domain_decomposition(f, L, 0.5, 500, 500, alpha_robin=1.0, filename='test', tol=1e-8)



    # _, errors = multi_subdomain_decomposition(f_interior_layer, L, [0.15, 0.42, 0.58, 0.85], [100, 100, 100, 100, 100], alpha_robin=2.0, filename='multiple_interior_layer', tol=1e-20)

    # multiple interior layers with oscillations
    # _, errors = multi_subdomain_decomposition(partial(f_interior_layer_multiple, period=1000), L, [0.15, 0.5, 0.85], [100, 1000, 1000, 100], alpha_robin=2.0, filename='multiple_interior_layer', tol=1e-20)
    # multiple interior layers with NO oscillations
    # _, errors = multi_subdomain_decomposition(f_interior_layer_multiple, L, [0.15, 0.5, 0.85], [100, 1000, 1000, 100], alpha_robin=2.0, filename='multiple_interior_layer', tol=1e-20)

    # Shows Interior Layer
    # _,_,_,_, errors = lions_robin_domain_decomposition(f_interior_layer, L, 0.45, 400, 600, alpha_robin=1.0, filename='interior_layer', tol=1e-20)
    # u1, u2, errors = lions_robin_domain_decomposition(g, L, 0.2, 100, 160, alpha_robin=0.4, filename='nonlipschitz')
    # u1, u2, errors = lions_robin_domain_decomposition(f, L, 0.23, N1, N2, alpha_robin=0.6, filename='nonoverlap_left')
    # u1, u2, errors = lions_robin_domain_decomposition(f, L, 0.83, 1000, 100, alpha_robin=0.6, filename='nonoverlap_right')
