import numpy as np
import matplotlib.pyplot as plt

def f_actual(x):
    """docstring for f_actual"""
    return -1*(1 / (9*np.pi*np.pi))*np.sin(3*np.pi * x)

def run_lions_domain_decomposition(f, L, alpha, N1, N2, eta=1.0, alpha_robin=1.0, tol=1e-16, max_iter=3000):
    """
    Runs the P.-L. Lions' domain decomposition method with Robin boundary conditions.

    Parameters:
    - f: function, source term function
    - L: float, length of the domain
    - alpha: float, shared boundary point between the two subdomains
    - N1: int, number of grid points in the first subdomain
    - N2: int, number of grid points in the second subdomain
    - eta: float, parameter for the differential equation (default 1.0)
    - alpha_robin: float, coefficient for the Robin boundary condition (default 1.0)
    - tol: float, tolerance for convergence (default 1e-6)
    - max_iter: int, maximum number of iterations (default 1000)

    Returns:
    - None, but plots the convergence rate and the final solution
    """

    # Set up the subdomains with different grid spacings
    x1,h1 = np.linspace(0, alpha, N1 + 1, retstep=True)
    x2,h2 = np.linspace(alpha, L, N2 + 1, retstep=True)
    # h1 = x1[1] - x1[0]  # Grid spacing for the first subdomain
    # h2 = x2[1] - x2[0]  # Grid spacing for the second subdomain

    # Initial guess for lambda values on the interface
    lambda1 = 0.0
    lambda2 = 0.0

    # Initial guess for the solution in each subdomain
    u0_1 = np.zeros(N1 + 1)
    u0_2 = np.zeros(N2 + 1)

    # Helper function to solve the subdomain problem with Robin boundary condition
    def solve_subdomain(f_values, u, h, g, alpha_robin,lhs=True):
        # Create a new array for the subdomain solution
        u_new = np.copy(u)
        
        # Solve the interior points of the subdomain
        if lhs:
            for i in range(1, len(u_new) - 1):
                u_new[i] = (u_new[i - 1] + u_new[i + 1] - h**2 * f_values[i]) / 2
            

            # u_new[-2] = 2*u_new[-3] - h**2 *f_values[-2] -  u_new[-4]
            u_new[-1] = (h*g + u_new[-2]) / (1 + alpha_robin * h)
            # u_new[-1] = 2*u_new[-2] + h**2 *u_prev +  u_new[-3]
        else:
            f_rev = np.flip(f_values)
            u_new =  np.flip(u_new)

            for i in range(1, len(u_new) - 1):
                u_new[i] = (u_new[i - 1] + u_new[i + 1] - h**2 * f_rev[i]) / 2
            
            # u_new[-2] = 2*u_new[-3] - h**2 *f_rev[-2] -  u_new[-4]
            u_new[-1] = (h*g + u_new[-2]) / (1 + alpha_robin * h)
            # u_new[-1] = (lambda_value + alpha_robin * u_prev)
            # u_new[-1] = 2*u_new[-2] - h**2 *u_prev -  u_new[-3]
            u_new = np.flip(u_new)

        # if lhs:
        #     lambda_new = (u_new[-1] - u_new[-2]) / h
        # else:
        #     lambda_new = -1*(u_new[-1] - u_new[-2]) / h
            

        
        return u_new

    # Main function implementing P.-L. Lions' algorithm with Robin boundary conditions
    def lions_domain_decomposition_robin(f, u0_1, u0_2, x1, x2, h1, h2, alpha_robin, tol, max_iter):
        u1, u2 = u0_1.copy(), u0_2.copy()
        errors = []
        
        # derivative of normals on bdy
        lambda1, lambda2 = 0.0, 0.0  # Initial lambda values
        g1, g2 = 0.0, 0.0
        
        for iter in range(max_iter):
            u1_old, u2_old = u1.copy(), u2.copy()
            
            # Solve on the first subdomain with lambda1 as the boundary condition
            f_values1 = f(x1)
            u1 = solve_subdomain(f_values1, u1, h1, g1, alpha_robin,lhs=True)
            
            # Solve on the second subdomain with lambda2 as the boundary condition
            f_values2 = f(x2)
            u2 = solve_subdomain(f_values2, u2, h2, g2, alpha_robin,lhs=False)

            print(iter)
            print(u2[0])
            print(u2[1])
            print(u1[-1])
            g1 = -1*((u2[0] - u2[1]) / h2) + alpha_robin * u2[0]
            g2 = -1*((u1[-1] - u1[-2]) / h1) + alpha_robin * u1[-1]
            
            # Update lambda values according to the transmission conditions
            lambda1_new = -lambda2 + 2 * alpha_robin * u2[0]
            lambda2_new = -lambda1 + 2 * alpha_robin * u1[-1]
            
            # Calculate error and check convergence
            error = np.linalg.norm(np.concatenate((u1, u2)) - np.concatenate((f_actual(x1), f_actual(x2))), ord=2)
            errors.append(error)
            
            if error < tol:
                break
            
            # Update lambdas for the next iteration
            lambda1, lambda2 = lambda1_new, lambda2_new
        
        return u1, u2, errors

    # Run the domain decomposition method
    u1, u2, errors = lions_domain_decomposition_robin(f, u0_1, u0_2, x1, x2, h1, h2, alpha_robin, tol, max_iter)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot the convergence rate on the first subplot
    ax1.plot(errors)
    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error (L2 norm)')
    ax1.set_title("Convergence of P.-L. Lions' Domain Decomposition Method")
    ax1.grid(True)

    # Plot the final solution on the second subplot
    ax2.plot(x1, u1, label='Subdomain 1 Solution')
    ax2.plot(x2, u2, label='Subdomain 2 Solution')
    ax2.plot(np.linspace(0,L,300), f_actual(np.linspace(0,L,300)))
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.set_title("Solution of the PDE using P.-L. Lions' Domain Decomposition")
    ax2.legend()
    ax2.grid(True)

    # Show the combined plot
    plt.tight_layout()
    plt.savefig('test.png')

# Example usage of the function
run_lions_domain_decomposition(
    f=lambda x: np.sin(3*np.pi * x),  # Source function
    L=1.0,                          # Length of the domain
    alpha=0.2,                      # Shared boundary point
    alpha_robin=0.25,
    N1=20,                          # Grid resolution for the first subdomain
    N2=80                          # Grid resolution for the second subdomain
)

