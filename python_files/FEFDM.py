# In this code we implement the forward Euler finite difference method. 

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
import time


def load_mnist_image(index=0):
    """Load a single MNIST image"""
    transform = transforms.ToTensor()
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    image, label = mnist_dataset[index]
    return image.squeeze().numpy(), label

def diffuse_image_heat_equation(image, alpha=0.1, num_steps=50, dt=0.1):
    """
    Diffuse an image using the heat equation with explicit finite differences

    Parameters:
    - image: 2D numpy array (input image)
    - alpha: diffusion coefficient
    - num_steps: number of time steps
    - dt: time step size

    Returns:
    - diffused_images: list of images at different time steps
    """
    u = image.copy().astype(np.float64)
    diffused_images = [u.copy()]

    dx, dy = 1.0, 1.0

    # Stability check
    stability_condition = alpha * dt * (1/dx**2 + 1/dy**2)
    if stability_condition > 0.5:
        print(f"Warning: Stability condition: {stability_condition:.3f}")

    for step in range(num_steps):
        laplacian = np.zeros_like(u)

        # Interior points (standard 5-point stencil)
        laplacian[1:-1, 1:-1] = (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )

        # Boundary points with second-order Neumann BCs
        # Top boundary (i=0): uses ghost cell u[-1,j] = u[1,j]
        laplacian[0, 1:-1] = (
            (u[1, 1:-1] - 2*u[0, 1:-1] + u[1, 1:-1]) / dx**2 +  
            (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) / dy**2
        )

        # Bottom boundary (i=-1): uses ghost cell u[N,j] = u[N-2,j]  
        laplacian[-1, 1:-1] = (
            (u[-2, 1:-1] - 2*u[-1, 1:-1] + u[-2, 1:-1]) / dx**2 +  
            (u[-1, 2:] - 2*u[-1, 1:-1] + u[-1, :-2]) / dy**2
        )

        # Left boundary (j=0): uses ghost cell u[i,-1] = u[i,1]
        laplacian[1:-1, 0] = (
            (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0]) / dx**2 +
            (u[1:-1, 1] - 2*u[1:-1, 0] + u[1:-1, 1]) / dy**2 
        )

        # Right boundary (j=-1): uses ghost cell u[i,N] = u[i,N-2]
        laplacian[1:-1, -1] = (
            (u[2:, -1] - 2*u[1:-1, -1] + u[:-2, -1]) / dx**2 +
            (u[1:-1, -2] - 2*u[1:-1, -1] + u[1:-1, -2]) / dy**2  
        )

        # Corners (use average of adjacent approaches)
        # Top-left (0,0)
        laplacian[0, 0] = (
            (u[1, 0] - 2*u[0, 0] + u[1, 0]) / dx**2 +
            (u[0, 1] - 2*u[0, 0] + u[0, 1]) / dy**2
        )

        # Top-right (0,-1)
        laplacian[0, -1] = (
            (u[1, -1] - 2*u[0, -1] + u[1, -1]) / dx**2 +
            (u[0, -2] - 2*u[0, -1] + u[0, -2]) / dy**2
        )

        # Bottom-left (-1,0)
        laplacian[-1, 0] = (
            (u[-2, 0] - 2*u[-1, 0] + u[-2, 0]) / dx**2 +
            (u[-1, 1] - 2*u[-1, 0] + u[-1, 1]) / dy**2
        )

        # Bottom-right (-1,-1)
        laplacian[-1, -1] = (
            (u[-2, -1] - 2*u[-1, -1] + u[-2, -1]) / dx**2 +
            (u[-1, -2] - 2*u[-1, -1] + u[-1, -2]) / dy**2
        )

        # Update using explicit Euler
        u = u + alpha * dt * laplacian
        diffused_images.append(u.copy())

    return diffused_images

# Example usage
if __name__ == "__main__":
    # Load MNIST image
    image, label = load_mnist_image(index=0)  # Try different indices
    print(f"Loaded MNIST digit: {label}")

    print("Running diffusion simulation...")
    start_time = time.time() 
    results = diffuse_image_heat_equation(image, alpha=0.1, num_steps=1000, dt=0.1)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    time_indices = [0, 100, 250, 500, 750, 900, 950, 999]

    for idx, ax in enumerate(axes.flat):
        if idx < len(time_indices):
            time_idx = time_indices[idx]
            ax.imshow(results[time_idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Step {time_idx}')
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle('MNIST Image Diffusion using Heat Equation', y=1.02)
    plt.show()

    # Plot intensity profile over time
    plt.figure(figsize=(10, 6))
    center_intensity = [img[14, 14] for img in results]  # Center pixel
    plt.plot(center_intensity)
    plt.xlabel('Time Step')
    plt.ylabel('Intensity at Center')
    plt.title('Intensity Evolution at Image Center')
    plt.grid(True)
    plt.show()

    print(image.shape)
