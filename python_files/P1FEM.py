import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class P1FEMDiffusion:
    """
    Simplified P1 Finite Element Method for image diffusion
    """

    def __init__(self, dt=0.1, steps=50):
        self.dt = dt
        self.steps = steps
        self._assembled = False

    def load_image(self, image):
        self.original_image = image.astype(float)
        self.nx, self.ny = self.original_image.shape
        self.N = self.nx * self.ny
        self.u = self.original_image.flatten().copy()

        self.hx = 1.0 / (self.nx - 1)
        self.hy = 1.0 / (self.ny - 1)
        self._assembled = False

    def _index_map(self, i, j):
        return i * self.ny + j

    def _create_mesh(self):
        self.coords = np.array([[i * self.hx, j * self.hy] 
                               for i in range(self.nx) for j in range(self.ny)])

        self.triangles = []
        for i in range(self.nx - 1): 
            for j in range(self.ny - 1): 
                n0 = self._index_map(i, j)
                n1 = self._index_map(i + 1, j)
                n2 = self._index_map(i, j + 1)
                n3 = self._index_map(i + 1, j + 1)
                self.triangles.append([n0, n1, n2])
                self.triangles.append([n1, n3, n2])

    def _assemble_matrices(self):
        if self._assembled:
            return

        self._create_mesh()

        K = lil_matrix((self.N, self.N)) 
        M = lil_matrix((self.N, self.N))

        for tri in self.triangles: 
            nodes = np.array([self.coords[i] for i in tri])

            v1 = nodes[1] - nodes[0]
            v2 = nodes[2] - nodes[0]
            area = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])

            mat = np.ones((3, 3))
            mat[:, 1:] = nodes 
            grads = np.linalg.inv(mat)[:, 1:]

            Ke = area * grads @ grads.T
            Me = (area / 12.0) * (np.ones((3, 3)) + np.eye(3))

            for a in range(3): 
                for b in range(3): 
                    K[tri[a], tri[b]] += Ke[a, b]
                    M[tri[a], tri[b]] += Me[a, b]

        self.K = K.tocsr()
        self.M = M.tocsr()
        self._assembled = True

    def _calculate_gradient_energy(self, u_array):
        u_img = u_array.reshape((self.nx, self.ny))
        grad_x = np.diff(u_img, axis=0)
        grad_y = np.diff(u_img, axis=1)
        return np.sum(grad_x**2) + np.sum(grad_y**2)

    def solve(self, convergence_tol=1e-6):
        if not self._assembled:
            self._assemble_matrices()

        # Store initial energy
        self.initial_energy = self._calculate_gradient_energy(self.u)
        self.energies = [self.initial_energy]

        for step in range(self.steps):
            A = self.M + self.dt * self.K
            b = self.M @ self.u

            u_new = spsolve(A, b)

            energy = self._calculate_gradient_energy(u_new)
            self.energies.append(energy)

            diff_norm = np.linalg.norm(u_new - self.u)
            self.u = u_new

            if diff_norm < convergence_tol:
                print(f"Converged at step {step}")
                break

        self.final_energy = energy
        return self.get_solution()

    def get_solution(self):
        return self.u.reshape((self.nx, self.ny))

    def plot_comparison(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Diffused image
        diffused = self.get_solution()
        axes[1].imshow(diffused, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Diffused Image")
        axes[1].axis('off')

        # Energy plot
        axes[2].plot(self.energies)
        axes[2].set_title("Gradient Energy")
        axes[2].set_xlabel("Time Step")
        axes[2].set_ylabel("Energy")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print energy information
        print(f"Initial energy: {self.initial_energy:.3f}")
        print(f"Final energy: {self.final_energy:.3f}")
        print(f"Energy reduction: {self.initial_energy/self.final_energy:.2f}x")


# Example usage
if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from torchvision import transforms

    # Load MNIST digit
    mnist = MNIST(root='.', train=False, download=True, transform=transforms.ToTensor())
    digit_tensor, _ = mnist[0]
    digit = digit_tensor.squeeze().numpy()

    # Create and run diffusion
    solver = P1FEMDiffusion(dt=0.1, steps=50)
    solver.load_image(digit)
    diffused = solver.solve()

    # Show results
    solver.plot_comparison()
