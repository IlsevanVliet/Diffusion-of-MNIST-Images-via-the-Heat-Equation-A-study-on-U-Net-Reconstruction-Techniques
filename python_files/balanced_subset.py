import torch 
from collections import defaultdict

def create_balanced_subset(dataset, samples_per_class=50, seed=42):
    """
    Create a balanced subset with equal samples per class
    """
    class_indices = defaultdict(list)

    # Group indices by class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    # Select balanced samples
    selected_indices = []
    for class_idx in range(10):  # 0-9 for MNIST
        class_samples = class_indices[class_idx]
        if len(class_samples) > samples_per_class:
            selected = torch.randperm(len(class_samples))[:samples_per_class]
            selected_indices.extend([class_samples[i] for i in selected])
        else:
            selected_indices.extend(class_samples)

    return torch.utils.data.Subset(dataset, selected_indices)
