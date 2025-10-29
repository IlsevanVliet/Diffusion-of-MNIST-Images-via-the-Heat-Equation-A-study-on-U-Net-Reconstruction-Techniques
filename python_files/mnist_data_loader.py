
# In this code we create the training and test dataloaders from the MNIST dataset, using the datasets of PyTorch.

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

def mnist_data_loader(): 
    # Load the data 
    train_data = datasets.MNIST(root="data", train=True, download=False, transform=transforms.Compose([transforms.Resize((32, 32)), ToTensor()]))
    test_data = datasets.MNIST(root="data", train=False, download=False, transform=transforms.Compose([transforms.Resize((32, 32)), ToTensor()]))

    # Create class names and class name to index mapping for MNIST 
    class_names = train_data.classes 
    class_names_to_idx = train_data.class_to_idx

    # Create data loaders 
    BATCH_SIZE = 32 
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_data, test_data, train_dataloader, test_dataloader, BATCH_SIZE
