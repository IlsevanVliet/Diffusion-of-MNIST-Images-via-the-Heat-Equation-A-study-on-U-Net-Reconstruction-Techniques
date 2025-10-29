def train_loop_precomputed(num_epochs, model, train_loader, test_loader, loss_fn, optimizer, scheduler, device): 
    model.to(device)
    train_losses = []
    test_losses = []
    epochs = []
    for epoch in range(num_epochs): 
        train_loss = 0.0
        model.train() 

        for batch, (X_noisy, X_clean, y) in enumerate(train_loader): 
            X_noisy, X_clean = X_noisy.to(device), X_clean.to(device)

            pred = model(X_noisy)
            loss = loss_fn(pred, X_clean) 

            optimizer.zero_grad() 
            loss.backward() 

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step() 

            train_loss += loss.item() 

            if batch % 100 == 0: 
                print(f"Epoch:{epoch+1}, batch: {batch}, Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # # Save checkpoints 
        # if (epoch + 1) % 5 == 0: 
        #     torch.save(model.state_dict(), f"checkpoints_epoch_{epoch+1}.pth")

        train_losses.append(train_loss)

        model.eval() 
        test_loss = 0 

        with torch.no_grad(): 
            for X_noisy, X_clean, y in test_loader: 
                X_noisy, X_clean = X_noisy.to(device), X_clean.to(device)

                pred = model(X_noisy)
                loss = loss_fn(pred, X_clean)
                test_loss += loss.item() 

        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        scheduler.step(test_loss)
        epochs.append(epoch)
        print(f"Test Loss: {test_loss:.4f}")

    plot_comparison_precomputed(X_clean, X_noisy, pred)


    # torch.save(model.state_dict(), "unet2_mnist.pth")
    print("Training completed!")
    return train_losses, test_losses

def test_loop_precomputed(model, test_loader, loss_fn, device): 
    model.eval() 
    test_loss = 0 
    test_losses = []
    with torch.no_grad(): 
        for X_noisy, X_clean, y in test_loader: 
            X_noisy, X_clean = X_noisy.to(device), X_clean.to(device)

            pred = model(X_noisy)
            loss = loss_fn(pred, X_clean)
            test_loss += loss.item() 

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    print(f"Test Loss: {test_loss:.4f}")

    plot_comparison_precomputed(X_clean, X_noisy, pred)
    return test_losses

def plot_comparison_precomputed(clean, noisy, pred, num_images=3): 
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4*num_images))

    if num_images ==1: 
        axes = axes.reshape(1, -1)

    for i in range(num_images): 
        axes[i,0].imshow(clean[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i,0].axis('off')

        axes[i,1].imshow(noisy[i].cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i,1].set_title('Diffused')
        axes[i,1].axis('off')

        axes[i,2].imshow(pred[i].cpu().squeeze(), cmap='gray')
        axes[i,2].set_title('Reconstructed')
        axes[i,2].axis('off')

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

def train_with_precomputed(num_epoch): 
    # Load precomputed datasets 
    train_dataset = PrecomputedDataset("mnist_train_diffused.pth")
    test_dataset = PrecomputedDataset("mnist_test_diffused.pth")

    # Create dataloaders 
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)

    # Model and optimizer 
    model = U_net2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=0, factor=0.1)
    loss_fn = nn.MSELoss() 

    train_losses, test_losses = train_loop_precomputed(num_epochs=num_epoch, model=model, train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, device=device)
    #test_losses = test_loop_precomputed(model, test_loader, loss_fn, device)

    plot_losses(train_losses, test_losses)
train_with_precomputed(10)


