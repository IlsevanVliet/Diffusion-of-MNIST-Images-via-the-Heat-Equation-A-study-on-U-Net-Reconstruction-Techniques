import torch 
import torch.nn as nn 
import standard_unet 
from standard_unet import U_net1
from Q1FEM import QuadElementFEM
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import balanced_subset 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model, loss function and optimizer
model = U_net1().to(device) 
fem_solver = QuadElementFEM(nx=31, ny=31, Lx=1.0, Ly=1.0)         
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) 

# Create consistent diffusion by going from tensor to numpy, diffuse and back to tensor 
def diffuse_batch_fem(image_batch, fem_solver, dt, n_steps, diffusion_coeff):
    """Apply diffusion to a batch of images using FEM"""
    batch_size = image_batch.shape[0]
    diffused_batch = []
    for i in range(batch_size): 
        img_np = image_batch[i].squeeze().cpu().numpy() # Convert to numpy and remove channel dimension 
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) # Normalize to [0,1]
        if img_np.shape != (32, 32): 
            img_np = img_np.reshape(32, 32)
        u0 = img_np.flatten() 
        if len(u0) > fem_solver.n_nodes:
            u0 = u0[:fem_solver.n_nodes]
        elif len(u0) < fem_solver.n_nodes:
            u0 = np.pad(u0, (0, fem_solver.n_nodes - len(u0)))
        diffused_results = fem_solver.diffuse(u0, dt, n_steps, diffusion_coeff)
        diffused_img = diffused_results[-1] # We are only interested in the final result 
        diffused_tensor = torch.from_numpy(diffused_img).float().unsqueeze(0)
        diffused_batch.append(diffused_tensor)

    results = torch.stack(diffused_batch).to(image_batch.device)

    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
    return results

dt = 0.1
n_steps = 10
diffusion_coeff = 0.1

# Create a function for the training loop 

def train_loop(num_epochs, model, balanced_dataloader, balanced_test_dataloader, loss_fn, optimizer, fem_solver, dt, n_steps, diffusion_coeff):
    train_losses = []
    test_losses = []
    epochs = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch, (X, y) in enumerate(balanced_dataloader): # This means the batch is the index of the batch, while (X,y) is the actual data
            X, y = X.to(device), y.to(device)
            X_noisy = diffuse_batch_fem(X, fem_solver, dt, n_steps, diffusion_coeff)
            # Forward pass
            pred = model(X_noisy)
            loss = loss_fn(pred, X) # We want to reconstruct the original image
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch % 20 == 0: 
                print(f"Batch:{batch}, Loss: {loss.item():.4f}")
                # Debug: check value ranges
                print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
                print(f"  X_noisy range: [{X_noisy.min():.3f}, {X_noisy.max():.3f}]")
                print(f"  Pred range: [{pred.min():.3f}, {pred.max():.3f}]")

        train_loss /= len(balanced_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        model.eval() 
        with torch.no_grad(): 
            test_loss = 0 
            for X,y in balanced_test_dataloader: 
                X, y = X.to(device), y.to(device)
                X_noisy = diffuse_batch_fem(X, fem_solver, dt, n_steps, diffusion_coeff)
                # X_noisy = X_noisy.to(device)

                # if X_noisy.dim() == 3: 
                #     X_noisy = X_noisy.unsqueeze(0) 

                pred = model(X_noisy)
                loss = loss_fn(pred, X)
                test_loss += loss.item() 


            test_loss /= len(balanced_test_dataloader)
            print(f"Test Loss: {test_loss:.4f}")
            test_losses.append(test_loss) 
            epochs.append(epoch)

    plot_comparison(X, X_noisy, pred, num_images = 3)
    return train_losses, test_losses


def plot_comparison(X, X_noisy, pred, num_images=3):
    """Comparison plot"""
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 3 * num_images))

    if num_images == 1: 
        axes = axes.reshape(1, -1)

    titles = ["Original Input", "Noisy Input", "Predicted output"]

    for i in range(num_images): 
        orig_img = X[i].cpu().squeeze()
        noisy_img = X_noisy[i].cpu().squeeze()
        pred_img = pred[i].cpu().squeeze() 

        axes[i,0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[i,1].imshow(noisy_img, cmap='gray', vmin=0, vmax=1)
        axes[i,2].imshow(pred_img, cmap='gray', vmin=0, vmax=1)

        axes[i,0].set_title(f"{titles[0]}\n[{orig_img.min():.2f}, {orig_img.max():.2f}]")
        axes[i,1].set_title(f"{titles[1]}\n[{noisy_img.min():.2f}, {noisy_img.max():.2f}]")
        axes[i,2].set_title(f"{titles[2]}\n[{pred_img.min():.2f}, {pred_img.max():.2f}]")

        for j in  range(3): 
            axes[i,j].axis('off')
    plt.tight_layout() 
    plt.show()

def plot_losses(train_losses, test_losses): 
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Training Loss', color = 'blue')
    plt.plot(test_losses, label='Testing Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss and Test Loss over epochs')
    plt.legend()
    plt.show() 
