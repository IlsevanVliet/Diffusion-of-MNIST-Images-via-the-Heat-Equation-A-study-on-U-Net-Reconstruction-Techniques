import torch 
from torch.utils.data import Dataset
from Q1FEM import QuadElementFEM
import mnist_data_loader 

# Pre-diffuse the entire dataset and save it 
def precompute_diffused_dataset(dataloader, fem_solver, dt, n_steps, diffusion_coeff, save_path): 
    all_diffused = []
    all_original = []
    all_labels = [] 
    for batch, (X,y) in enumerate(dataloader): 
        print(f"Precomputing batch {batch}/{len(dataloader)}")
        X_diffused = QuadElementFEM.diffuse_batch_fem(X, fem_solver, dt, n_steps, diffusion_coeff)
        all_diffused.append(X_diffused.cpu())
        all_original.append(X.cpu()) 
        all_labels.append(y.cpu())

    diffused_tensor = torch.cat(all_diffused)
    original_tensor = torch.cat(all_original)
    labels_tensor = torch.cat(all_labels)

    torch.save({'diffused': diffused_tensor, 
                'original': original_tensor, 
                'labels': labels_tensor}, save_path)
    return diffused_tensor, original_tensor, labels_tensor

class PrecomputedDataset(Dataset): 
    def  __init__(self, data_path): 
        data = torch.load(data_path)
        self.diffused = data['diffused']
        self.original = data['original']
        self.labels = data['labels']

        #print(f"Initial shapes - Diffused: {self.diffused.shape}, Original: {self.original.shape}")

        if self.diffused.dim() == 3: 
            self.diffused = self.diffused.unsqueeze(1)
        if self.original.dim() == 3:
            self.original = self.original.unsqueeze(1)

        print(f"Final shapes - Diffused: {self.diffused.shape}, Original: {self.original.shape}")

        # Verify data ranges
        print(f"Value ranges - Original: [{self.original.min():.3f}, {self.original.max():.3f}]")
        print(f"Value ranges - Diffused: [{self.diffused.min():.3f}, {self.diffused.max():.3f}]")

        # Normalize if needed
        self._normalize_data()

    def _normalize_data(self):
        """Normalize data to [0, 1] range if needed"""
        if self.original.max() > 1.0 or self.original.min() < 0.0:
            self.original = (self.original - self.original.min()) / (self.original.max() - self.original.min())
            print("Normalized original images to [0, 1]")

        if self.diffused.max() > 1.0 or self.diffused.min() < 0.0:
            self.diffused = (self.diffused - self.diffused.min()) / (self.diffused.max() - self.diffused.min())
            print("Normalized diffused images to [0, 1]")

    def __len__(self): 
        return len(self.diffused)

    def __getitem__(self, idx): 
        return self.diffused[idx], self.original[idx], self.labels[idx]

# Run precomputation for the diffusion of the images 

def run_precomputation(): 
    train_data, test_data, train_dataloader, test_dataloader, BATCH_SIZE = mnist_data_loader() 

    fem_solver = QuadElementFEM(nx=31, ny=31, Lx=1.0, Ly=1.0)

    dt = 0.1
    n_steps = 10 
    diffusion_coeff = 0.1

    print("Precomputing training dataset")
    precompute_diffused_dataset(train_dataloader, fem_solver, dt, n_steps, diffusion_coeff, "mnist_train_diffused.pth")

    print("Precomputing test dataset")
    precompute_diffused_dataset(test_dataloader, fem_solver, dt, n_steps, diffusion_coeff, "mnist_test_diffused.pth")
