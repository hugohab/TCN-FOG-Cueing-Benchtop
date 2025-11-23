import torch
from torch.utils.data import DataLoader # handles batching, shuffling, and parallel data loading
from sklearn.model_selection import train_test_split # makes train and test sets
from src_alex.dataset import FOGDataset # loads/normalizes data
from src_alex.model import FOGTCN # is the neural network
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt


def train(npz_path, epochs=25, batch_size=32, lr=1e-3,debug_single_file=False):

    dataset = FOGDataset(npz_path,debug_single_file) # Reads data, normalizes it, and makes it PyTorch-ready.

    # OPTIONAL: speed up debugging
    if debug_single_file:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(1000))  # only use first 1000 sample

    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, shuffle=True) # → 80% training, 20% held-out testing.this is a random split, not subject-based.

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx),num_workers=2, pin_memory=True) # DataLoader makes training efficient by: batching data, shuffling samples, moving data to GPU easily later (if needed)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx),num_workers=2, pin_memory=True)

     # ---- MOVE MODEL TO GPU ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = FOGTCN().to(device)

    loss_fn = nn.CrossEntropyLoss() # used for classification, teaches model how wrong it is # weight penalize fog when its no fog an dother way around
    # create a function that calculates weights to implement in loss function
    optimizer = optim.Adam(model.parameters(), lr=lr) # optimizer updates learned weights to reduce loss

    loss_history = []  # <— list to store loss

    for epoch in range(epochs): # This runs over all epochs:
        model.train() 
        total_loss = 0

        for x, y in train_loader:

            # ---- MOVE DATA TO GPU ----
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x) # model predicts
            loss = loss_fn(logits, y) # compare prediction vs label
            loss.backward() # compute gradients
            optimizer.step() # update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)  # store loss

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    print("Dataset size:", len(dataset))


     # ---- Plot Loss Curve ----
    plt.figure(figsize=(6,4))
    plt.plot(loss_history, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()

    torch.save(model.state_dict(), "fog_tcn.pth") # Save the trained model

    return model, test_loader


