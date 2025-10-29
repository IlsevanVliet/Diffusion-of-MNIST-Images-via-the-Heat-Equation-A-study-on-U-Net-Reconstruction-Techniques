
# In this code, we implement the backward Euler finite difference method.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
import torch
import torchvision
from torchvision import datasets, transforms

class HeatEquation2D:
    def __init__(self, alpha=0.1, dx=1.0, dy=1.0):
        self.alpha = alpha
        self.dx = dx
        self.dy = dy

    def create_laplacian_matrix(self, nx, ny):
        """Create the discrete Laplacian matrix with Neumann boundary conditions"""
        N = nx * ny
        # Main diagonal: -4 for interior points
        main_diag = -2 * (1/self.dx**2 + 1/self.dy**2) * np.ones(N)

        # x-direction neighbors (left and right)
        x_diag = (1/self.dx**2) * np.ones(N)
        # Remove connections at boundaries in x-direction
        for i in range(ny):
            x_diag[i * nx] = 0  # Left boundary
            x_diag[i * nx + nx - 1] = 0  # Right boundary

        # y-direction neighbors (above and below)
        y_diag = (1/self.dy**2) * np.ones(N)
        # Remove connections at boundaries in y-direction
        for i in range(nx):
            y_diag[i] = 0  # Top boundary
            y_diag[(ny - 1) * nx + i] = 0  # Bottom boundary

        # Create the sparse matrix
        diagonals = [main_diag, x_diag, x_diag, y_diag, y_diag]
        offsets = [0, -1, 1, -nx, nx]

        A = diags(diagonals, offsets, shape=(N, N), format='csr')
        return A

    def backward_euler_step(self, u, dt, A):
        """Perform one Backward Euler step"""
        N = u.size
        I = eye(N, format='csr')

        # Solve (I - α*dt*A) u^{n+1} = u^n
        lhs_matrix = I - self.alpha * dt * A
        u_flat = u.flatten()

        # Solve the linear system
        u_new_flat = spsolve(lhs_matrix, u_flat)
        u_new = u_new_flat.reshape(u.shape)

        return u_new

    def diffuse_image(self, u0, dt, num_steps):
        """Diffuse the initial image over time"""
        ny, nx = u0.shape
        A = self.create_laplacian_matrix(nx, ny)

        u = u0.copy()
        results = [u.copy()]

        for step in range(num_steps):
            u = self.backward_euler_step(u, dt, A)
            results.append(u.copy())

            # if (step + 1) % 10 == 0:
            #     print(f"Step {step + 1}/{num_steps}")

        return np.array(results)

def load_mnist_image():
    """Load and preprocess an MNIST image"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Use test dataset to avoid downloading training data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Get a digit image (e.g., the first one)
    image, label = test_dataset[0]

    # Convert to numpy and remove channel dimension
    image_np = image.squeeze().numpy()

    # Normalize to [0, 1]
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    return image_np, label

def visualize_results(initial_image, diffused_images, timesteps_to_show=None):
    """Visualize the diffusion process"""
    if timesteps_to_show is None:
        timesteps_to_show = [0, len(diffused_images)//4, len(diffused_images)//2, 
                           3*len(diffused_images)//4, len(diffused_images)-1]

    n_plots = len(timesteps_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 3))

    if n_plots == 1:
        axes = [axes]

    for i, timestep in enumerate(timesteps_to_show):
        axes[i].imshow(diffused_images[timestep], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Time step {timestep}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_energy_decay(diffused_images, dt):
    """Plot the decay of total energy over time"""
    energy = [np.sum(img**2) for img in diffused_images]
    time = np.arange(len(energy)) * dt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time, energy)
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Energy Decay Over Time')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(time, energy)
    plt.xlabel('Time')
    plt.ylabel('Total Energy (log scale)')
    plt.title('Energy Decay (Log Scale)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Parameters
    alpha = 0.1      # Diffusion coefficient
    dt = 0.1         # Time step
    num_steps = 1000   # Number of time steps

    # Load MNIST image
    print("Loading MNIST image...")
    initial_image, label = load_mnist_image()
    # print(f"Loaded digit: {label}")
    # print(f"Image shape: {initial_image.shape}")

    # Initialize heat equation solver
    heat_eq = HeatEquation2D(alpha=alpha)

    # Perform diffusion
    print("Starting diffusion process...")
    results = heat_eq.diffuse_image(initial_image, dt, num_steps)

    # Visualize results
    print("Visualizing results...")
    visualize_results(initial_image, results)

    # Plot energy decay
    plot_energy_decay(results, dt)

    # Print some statistics
    print(f"\nInitial energy: {np.sum(initial_image**2):.4f}")
    print(f"Final energy: {np.sum(results[-1]**2):.4f}")
    print(f"Energy conservation ratio: {np.sum(results[-1]**2)/np.sum(initial_image**2):.4f}")


