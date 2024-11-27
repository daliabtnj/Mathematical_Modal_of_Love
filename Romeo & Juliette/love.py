import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system of differential equations for Model I and II
def model_I(z, t):
    R, J = z
    dRdt = -3*R - 2*J
    dJdt = -2*R - 3*J
    return [dRdt, dJdt]

def model_II(z, t):
    R, J = z
    dRdt = -R + 4*J
    dJdt = 4*R - J
    return [dRdt, dJdt]

# Eigenvalues and eigenvectors for Model I
def eigenvalues_eigenvectors_A1():
    A = np.array([[-3, -2], [-2, -3]])
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs

# Eigenvalues and eigenvectors for Model II
def eigenvalues_eigenvectors_A2():
    A = np.array([[-1, 4], [4, -1]])
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs

# Solve the system of equations for Model I and II
def solve_system(model, initial_conditions, t):
    solution = odeint(model, initial_conditions, t)
    return solution

# Phase portrait function
def plot_phase_portrait(model, x_range, y_range, grid_density=20):
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], grid_density),
                       np.linspace(y_range[0], y_range[1], grid_density))
    U, V = np.zeros(X.shape), np.zeros(Y.shape)
    
    for i in range(len(X)):
        for j in range(len(Y)):
            dxdt, dydt = model([X[i, j], Y[i, j]], 0)  # Calculate derivatives at each grid point
            U[i, j] = dxdt
            V[i, j] = dydt

    plt.streamplot(X, Y, U, V, color='black', linewidth=1)
    plt.xlabel('R (Romeo)')
    plt.ylabel('J (Juliet)')
    plt.title('Phase Portrait')
    plt.show()

# Time range for simulation
t = np.linspace(0, 10, 500)

# Initial conditions for the models
initial_conditions = [1, 1]  # Initial values for R and J

# Solve the systems
solution_I = solve_system(model_I, initial_conditions, t)
solution_II = solve_system(model_II, initial_conditions, t)

# Plot phase portraits
plot_phase_portrait(model_I, [-3, 3], [-3, 3])  # Phase portrait for Model I
plot_phase_portrait(model_II, [-3, 3], [-3, 3])  # Phase portrait for Model II

# Plot the solution over time
plt.plot(t, solution_I[:, 0], label="R (Model I)")
plt.plot(t, solution_I[:, 1], label="J (Model I)")
plt.xlabel('Time')
plt.ylabel('Love between them')
plt.title('Solution for Model I (Romeo and Juliet)')
plt.legend()
plt.show()

plt.plot(t, solution_II[:, 0], label="R (Model II)")
plt.plot(t, solution_II[:, 1], label="J (Model II)")
plt.xlabel('Time')
plt.ylabel('Love between them')
plt.title('Solution for Model II (Romeo and Juliet)')
plt.legend()
plt.show()

# Eigenvalues and eigenvectors for both models
eigvals_I, eigvecs_I = eigenvalues_eigenvectors_A1()
eigvals_II, eigvecs_II = eigenvalues_eigenvectors_A2()

print("Eigenvalues and Eigenvectors for Model I:")
print("Eigenvalues:", eigvals_I)
print("Eigenvectors:", eigvecs_I)

print("\nEigenvalues and Eigenvectors for Model II:")
print("Eigenvalues:", eigvals_II)
print("Eigenvectors:", eigvecs_II)
