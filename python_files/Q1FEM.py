
# In this code, we implement a 2D heat equation to diffuse an image using the Finite Element Method with quadrilateral elements. 

import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torchvision
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time 
import os
from datetime import datetime 

class QuadElementFEM: 
    def __init__(self, nx, ny, Lx=1.0, Ly=1.0): 
        self.nx = nx 
        self.ny = ny 
        self.Lx = Lx 
        self.Ly = Ly 
        self.dx = Lx / nx 
        self.dy = Ly / ny
        self.n_nodes = (nx + 1) * (ny + 1)

        # 2x2 Gauss quadrature points and weights
        self.gauss_points = np.array([
            [-1/np.sqrt(3), -1/np.sqrt(3)], 
            [1/np.sqrt(3), -1/np.sqrt(3)], 
            [1/np.sqrt(3), 1/np.sqrt(3)], 
            [-1/np.sqrt(3), 1/np.sqrt(3)]  
        ])
        self.gauss_weights = np.ones(4)

        self._assemble_system()

    def basis_functions(self, xi, eta): 
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta), 
            0.25 * (1 - xi) * (1 + eta)
        ])
        return N

    def basis_derivatives(self, xi, eta): 
        dN_dxi = np.array([
            -0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)
        ])
        dN_deta = np.array([
            -0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)
        ])
        return dN_dxi, dN_deta

    def jacobian(self, nodes, xi, eta): 
        dN_dxi, dN_deta = self.basis_derivatives(xi, eta)

        dx_dxi = np.dot(dN_dxi, nodes[:, 0])
        dx_deta = np.dot(dN_deta, nodes[:, 0])
        dy_dxi = np.dot(dN_dxi, nodes[:, 1])
        dy_deta = np.dot(dN_deta, nodes[:, 1])

        J = np.array([[dx_dxi, dx_deta], 
                      [dy_dxi, dy_deta]])

        detJ = np.linalg.det(J)
        return J, detJ

    def _assemble_system(self): 
        self.K = lil_matrix((self.n_nodes, self.n_nodes))
        self.M = lil_matrix((self.n_nodes, self.n_nodes))

        for i in range(self.nx): 
            for j in range(self.ny): 
                node0 = j * (self.nx + 1) + i                   # left upper corner  
                node1 = j * (self.nx + 1) + i + 1               # right upper corner
                node2 = (j + 1) * (self.nx + 1) + i + 1         # right bottom corner  
                node3 = (j + 1) * (self.nx + 1) + i             # left bottom corner
                nodes = np.array([node0, node1, node2, node3])  # The four corners of each element

                coords = np.array([
                    [i * self.dx, j * self.dy], 
                    [(i + 1) * self.dx, j * self.dy], 
                    [(i + 1) * self.dx, (j + 1) * self.dy],
                    [i * self.dx, (j + 1) * self.dy]
                ])                                              # coordinates of the four corners of the element in the global domain on [0, Lx] x [0, Ly]

                Ke = np.zeros((4, 4))                           # Initialize element stiffness matrix
                Me = np.zeros((4, 4))                           # Initialize element mass matrix

                for gp, (xi, eta) in enumerate(self.gauss_points): 
                    weight = self.gauss_weights[gp]
                    N = self.basis_functions(xi, eta)
                    dN_dxi, dN_deta = self.basis_derivatives(xi, eta)

                    J, detJ = self.jacobian(coords, xi, eta)

                    if detJ <= 1e-12:
                        print(f"Warning: Small or negative Jacobian at element ({i},{j})")
                        continue

                    invJ = np.linalg.inv(J)

                    # Derivatives in global coordinates 
                    dN_dx = invJ[0,0] * dN_dxi + invJ[0, 1] * dN_deta
                    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

                    for a in range(4): 
                        for b in range(4): 
                            Ke[a, b] += (dN_dx[a] * dN_dx[b] + dN_dy[a] * dN_dy[b]) * detJ * weight
                            Me[a, b] += N[a] * N[b] * detJ * weight

                for a, node_a in enumerate(nodes): 
                    for b, node_b in enumerate(nodes): 
                        self.K[node_a, node_b] += Ke[a, b]
                        self.M[node_a, node_b] += Me[a, b]

        self.K = self.K.tocsr() 
        self.M = self.M.tocsr()

    def diffuse(self, u0, dt, n_steps, diffusion_coeff=1.0):
        """
        Diffuse initial condition using Backward Euler method
        """
        u = u0.copy()

        # System matrix for Backward Euler in time and finite element method space
        A = self.M + diffusion_coeff * dt * self.K

        results = [u.reshape(self.ny + 1, self.nx + 1)]

        for step in range(n_steps):
            # Right-hand side
            b = self.M.dot(u)

            u = spsolve(A, b)

            results.append(u.reshape(self.ny + 1, self.nx + 1))

        return np.array(results)

def load_mnist_sample(): 
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((32,32))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    image, label = trainset[0]
    image_np = image.squeeze().numpy()
    return image_np, label


def main(): 
    # Parameters 
    nx, ny = 31, 31  # If we have 27 elements we have 28 nodes, which matches MNIST
    dt = 0.1
    n_steps = 100
    diffusion_coeff = 0.1 

    print("Loading MNIST sample...")
    mnist_image, label = load_mnist_sample() 
    print(f"Loaded MNIST digit: {label}")

    print("Preprocessing image...")
    initial_condition = (mnist_image - mnist_image.min()) / (mnist_image.max() - mnist_image.min() + 1e-8)

    # Create FEM solver 
    print("Initializing FEM solver...")
    fem_solver = QuadElementFEM(nx, ny, Lx=1.0, Ly=1.0)

    # Flatten initial condition 
    u0 = initial_condition.flatten()

    # Check if sizes match
    print(f"Initial condition size: {len(u0)}, FEM nodes: {fem_solver.n_nodes}")

    if len(u0) != fem_solver.n_nodes: 
        print("Warning: Size mismatch, truncating...")
        u0 = u0[fem_solver.n_nodes]

    print("Running diffusion simulation...")
    start_time = time.time() 
    results = fem_solver.diffuse(u0, dt, n_steps, diffusion_coeff)
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds")


    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    time_indices = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]

    axes[0,0].imshow(initial_condition, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title('Original MNIST Image')
    axes[0,0].axis('off')

    for i, idx in enumerate(time_indices): 
        row = (i + 1) // 3 
        col = (i + 1) % 3 
        im = axes[row, col].imshow(results[idx], cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title(f'Time step {idx}')
        axes[row, col].axis('off')

    plt.tight_layout() 
    plt.show()

if __name__ == "__main__": 
    main()

