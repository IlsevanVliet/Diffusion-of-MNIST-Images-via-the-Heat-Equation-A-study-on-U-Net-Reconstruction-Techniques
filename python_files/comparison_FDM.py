
# In this code, we compare the forward Euler, backward Euler and Crank-Nicolson scheme in finite differences

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import torch
import torchvision
from torchvision import datasets, transforms
from matplotlib.animation import FuncAnimation

class HeatEquationAllMethods:
    def __init__(self, alpha=0.1, dx=1.0, dy=1.0):
        self.alpha = alpha
        self.dx = dx
        self.dy = dy

    def apply_neumann_bc(self, u):
        """Apply homogeneous Neumann boundary conditions"""
        u_new = u.copy()
        # Top and bottom
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]
        # Left and right  
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]
        return u_new

    # ================== EXPLICIT EULER ==================
    def explicit_euler_step(self, u, dt):
        """Explicit Euler - simplest implementation"""
        u_new = u.copy()

        # 5-point stencil for interior points
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + self.alpha * dt * (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dy**2
        )

        return self.apply_neumann_bc(u_new)

    def diffuse_explicit(self, u0, dt, num_steps):
        """Diffuse using Explicit Euler"""
        u = u0.copy()
        results = [u0.copy()]
        energies = [np.sum(u0**2)]

        for step in range(num_steps):
            u = self.explicit_euler_step(u, dt)
            results.append(u.copy())
            energies.append(np.sum(u**2))

        return np.array(results), np.array(energies)

    # ================== IMPLICIT EULER ==================
    def build_laplacian_matrix(self, nx, ny):
        """Build Laplacian matrix with Neumann BCs"""
        N = nx * ny
        main_diag = -2 * (1/self.dx**2 + 1/self.dy**2) * np.ones(N)
        x_diag = (1/self.dx**2) * np.ones(N)
        y_diag = (1/self.dy**2) * np.ones(N)

        # Remove connections at boundaries
        for i in range(ny):
            x_diag[i * nx] = 0  # Left boundary
            x_diag[i * nx + nx - 1] = 0  # Right boundary
        for i in range(nx):
            y_diag[i] = 0  # Top boundary
            y_diag[(ny - 1) * nx + i] = 0  # Bottom boundary

        A = diags([main_diag, x_diag, x_diag, y_diag, y_diag],
                 [0, -1, 1, -nx, nx], shape=(N, N), format='csr')
        return A

    def implicit_euler_step(self, u, dt):
        """Implicit Euler step"""
        ny, nx = u.shape
        A = self.build_laplacian_matrix(nx, ny)
        I = eye(nx * ny, format='csr')

        u_flat = u.flatten()
        u_new_flat = spsolve(I - self.alpha * dt * A, u_flat)
        u_new = u_new_flat.reshape(u.shape)

        return self.apply_neumann_bc(u_new)

    def diffuse_implicit(self, u0, dt, num_steps):
        """Diffuse using Implicit Euler"""
        u = u0.copy()
        results = [u0.copy()]
        energies = [np.sum(u0**2)]

        for step in range(num_steps):
            u = self.implicit_euler_step(u, dt)
            results.append(u.copy())
            energies.append(np.sum(u**2))

        return np.array(results), np.array(energies)

    # ================== CRANK-NICOLSON ==================
    def crank_nicolson_step(self, u, dt):
        """Crank-Nicolson step"""
        ny, nx = u.shape
        A = self.build_laplacian_matrix(nx, ny)
        I = eye(nx * ny, format='csr')

        lhs_matrix = I - 0.5 * self.alpha * dt * A
        rhs_matrix = I + 0.5 * self.alpha * dt * A

        u_flat = u.flatten()
        rhs_vec = rhs_matrix.dot(u_flat)
        u_new_flat = spsolve(lhs_matrix, rhs_vec)
        u_new = u_new_flat.reshape(u.shape)

        return self.apply_neumann_bc(u_new)

    def diffuse_crank_nicolson(self, u0, dt, num_steps):
        """Diffuse using Crank-Nicolson"""
        u = u0.copy()
        results = [u0.copy()]
        energies = [np.sum(u0**2)]

        for step in range(num_steps):
            u = self.crank_nicolson_step(u, dt)
            results.append(u.copy())
            energies.append(np.sum(u**2))

        return np.array(results), np.array(energies)

# ================== CORRECTED VISUALIZATION FUNCTIONS ==================

def load_mnist_image():
    """Load and preprocess an MNIST image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image, label = test_dataset[0]

    image_np = image.squeeze().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    return image_np, label

def compare_methods_side_by_side(initial_image, dt=0.1, num_steps=50):
    """Compare all three methods side by side"""
    solver = HeatEquationAllMethods(alpha=0.1)

    print("Running Explicit Euler...")
    results_explicit, energy_explicit = solver.diffuse_explicit(initial_image, dt, num_steps)

    print("Running Implicit Euler...")
    results_implicit, energy_implicit = solver.diffuse_implicit(initial_image, dt, num_steps)

    print("Running Crank-Nicolson...")
    results_cn, energy_cn = solver.diffuse_crank_nicolson(initial_image, dt, num_steps)

    return (results_explicit, energy_explicit, 
            results_implicit, energy_implicit, 
            results_cn, energy_cn)

def plot_comparison_evolution(results_dict, timesteps_to_show=None):
    """Plot evolution of all methods at selected timesteps"""
    if timesteps_to_show is None:
        n_steps = len(next(iter(results_dict.values()))[0])
        timesteps_to_show = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]

    methods = list(results_dict.keys())
    n_methods = len(methods)
    n_times = len(timesteps_to_show)

    fig, axes = plt.subplots(n_methods, n_times, figsize=(15, 3*n_methods))

    if n_methods == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        results = results_dict[method][0]
        for j, timestep in enumerate(timesteps_to_show):
            ax = axes[i][j] if n_methods > 1 else axes[j]
            im = ax.imshow(results[timestep], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{method}\nt={timestep}')
            ax.axis('off')

            # Add colorbar for each row
            if j == n_times - 1:
                plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()

def plot_energy_comparison(energy_dict, dt):
    """Plot energy decay comparison for all methods"""
    plt.figure(figsize=(12, 4))

    # Linear scale
    plt.subplot(1, 2, 1)
    for method, (_, energy) in energy_dict.items():
        time = np.arange(len(energy)) * dt
        plt.plot(time, energy, label=method, linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Energy Decay Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Log scale
    plt.subplot(1, 2, 2)
    for method, (_, energy) in energy_dict.items():
        time = np.arange(len(energy)) * dt
        plt.semilogy(time, energy, label=method, linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Total Energy (log scale)')
    plt.title('Energy Decay (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_method_specific_analysis(results_dict, dt):
    """CORRECTED: Method-specific analysis and visualization"""
    methods = list(results_dict.keys())

    plt.figure(figsize=(15, 5))

    # Maximum value decay
    plt.subplot(1, 3, 1)
    for method in methods:
        results = results_dict[method][0]
        max_vals = [np.max(frame) for frame in results]
        time = np.arange(len(max_vals)) * dt
        plt.plot(time, max_vals, label=method, linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Maximum Value')
    plt.title('Maximum Value Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Minimum value evolution
    plt.subplot(1, 3, 2)
    for method in methods:
        results = results_dict[method][0]
        min_vals = [np.min(frame) for frame in results]
        time = np.arange(len(min_vals)) * dt
        plt.plot(time, min_vals, label=method, linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Minimum Value')
    plt.title('Minimum Value Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # CORRECTED: Gradient magnitude calculation
    plt.subplot(1, 3, 3)
    for method in methods:
        results = results_dict[method][0]
        gradients = []
        for frame in results:
            # np.gradient returns (grad_y, grad_x) for 2D arrays
            grad_y, grad_x = np.gradient(frame)
            # Calculate magnitude at each point and sum
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            total_gradient = np.sum(grad_magnitude)
            gradients.append(total_gradient)

        time = np.arange(len(gradients)) * dt
        plt.plot(time, gradients, label=method, linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Total Gradient Magnitude')
    plt.title('Smoothing Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_simple_comparison(initial_image, dt=0.1, num_steps=30):
    """Simple comparison for quick results"""
    solver = HeatEquationAllMethods(alpha=0.1)

    # Run methods
    results_exp, energy_exp = solver.diffuse_explicit(initial_image, dt, num_steps)
    results_imp, energy_imp = solver.diffuse_implicit(initial_image, dt, num_steps)
    results_cn, energy_cn = solver.diffuse_crank_nicolson(initial_image, dt, num_steps)

    # Simple visualization
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))

    methods = ['Explicit Euler', 'Implicit Euler', 'Crank-Nicolson']
    all_results = [results_exp, results_imp, results_cn]

    timesteps = [0, num_steps//3, 2*num_steps//3, num_steps]

    for i, (method, results) in enumerate(zip(methods, all_results)):
        for j, t in enumerate(timesteps):
            im = axes[i, j].imshow(results[t], cmap='gray', vmin=0, vmax=1)
            axes[i, j].set_title(f'{method}\nTime: {t}')
            axes[i, j].axis('off')

            # Add colorbar for last column
            if j == len(timesteps) - 1:
                plt.colorbar(im, ax=axes[i, j], fraction=0.046)

    plt.tight_layout()
    plt.show()

    # # Energy comparison
    # plt.figure(figsize=(10, 4))
    # time = np.arange(len(energy_exp)) * dt

    # plt.subplot(1, 2, 1)
    # plt.plot(time, energy_exp, label='Explicit Euler', linewidth=2)
    # plt.plot(time, energy_imp, label='Implicit Euler', linewidth=2)
    # plt.plot(time, energy_cn, label='Crank-Nicolson', linewidth=2)
    # plt.xlabel('Time')
    # plt.ylabel('Total Energy')
    # plt.title('Energy Decay')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # plt.subplot(1, 2, 2)
    # plt.semilogy(time, energy_exp, label='Explicit Euler', linewidth=2)
    # plt.semilogy(time, energy_imp, label='Implicit Euler', linewidth=2)
    # plt.semilogy(time, energy_cn, label='Crank-Nicolson', linewidth=2)
    # plt.xlabel('Time')
    # plt.ylabel('Total Energy (log scale)')
    # plt.title('Energy Decay (Log Scale)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()

    return results_exp, results_imp, results_cn

# ================== MAIN EXECUTION ==================

def main():
    # Parameters
    dt = 0.1
    num_steps = 1000  # Reduced for faster execution

    # Load MNIST image
    print("Loading MNIST image...")
    initial_image, label = load_mnist_image()

    # Use simple comparison
    print("\nRunning comparison...")
    results_exp, results_imp, results_cn = create_simple_comparison(initial_image, dt, num_steps)

    # Organize results for additional analysis
    results_dict = {
        'Explicit Euler': (results_exp, np.array([np.sum(frame**2) for frame in results_exp])),
        'Implicit Euler': (results_imp, np.array([np.sum(frame**2) for frame in results_imp])),
        'Crank-Nicolson': (results_cn, np.array([np.sum(frame**2) for frame in results_cn]))
    }

    # # Run corrected analysis
    # print("\nRunning detailed analysis...")
    # plot_method_specific_analysis(results_dict, dt)

    # Print statistics
    print("\nFinal Statistics:")
    print("-" * 50)
    for method, (results, energy) in results_dict.items():
        initial_energy = energy[0]
        final_energy = energy[-1]
        conservation_ratio = final_energy / initial_energy
        print(f"{method:15} | Initial: {initial_energy:.4f} | Final: {final_energy:.4f} | Ratio: {conservation_ratio:.4f}")

if __name__ == "__main__":
    main()
